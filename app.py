import streamlit as st
from PIL import Image
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import vectorisation as vec
import plotly.express as px
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, HoverTool
from bokeh.transform import factor_cmap
import pandas as pd
import io
import os
from scipy.spatial import distance
from ripser import ripser
import persim
import landscapes as landscapes


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    gray_img = img.convert("L")
    return gray_img

@st.cache
def load_csv(csv_file):
    df = pd.read_csv(csv_file, names=['x', 'y', 'z'])
    return df

def infty_proj(x):
     return (256 if ~np.isfinite(x) else x)

@st.cache
def GetPds(data, isPointCloud):
    pd0 = pd1 = None

    if isPointCloud:
        data = data.to_numpy()
        dist_mat = distance.squareform(distance.pdist(data))
        ripser_result = ripser(dist_mat, maxdim=1, distance_matrix=True)
        pd0 = ripser_result['dgms'][0]
        pd1 = ripser_result['dgms'][1]

        pd0 = pd0[~np.isinf(pd0).any(axis=1),:]
        pd1 = pd1[~np.isinf(pd1).any(axis=1),:]
    else:
        data = np.array(data)
        data_gudhi = np.resize(data, [128, 128])
        data_gudhi = data_gudhi.reshape(128*128,1)
        cub_filtration = gd.CubicalComplex(dimensions = [128,128], top_dimensional_cells=data_gudhi)
        cub_filtration.persistence()

        pd0 = cub_filtration.persistence_intervals_in_dimension(0)
        pd1 = cub_filtration.persistence_intervals_in_dimension(1)

        for j in range(pd0.shape[0]):
            if pd0[j,1]==np.inf:
                pd0[j,1]=256
        for j in range(pd1.shape[0]):
            if pd1[j,1]==np.inf:
                pd1[j,1]=256
    
    return pd0, pd1

def CreateDownloadButton(label, data):
    # Create an in-memory buffer
    with io.BytesIO() as buffer:
        # Write array to buffer
        for row in data:
            np.savetxt(buffer, [row], fmt='%5d',delimiter=',') 
        st.download_button(
            label=f"Download {label}",
            data = buffer, # Download buffer
            file_name = f'{label}.csv',
            mime='text/csv')

def ApplyHatchPatternToChart(fig):
    fig.ygrid.grid_line_color = None
    fig.xgrid.band_hatch_pattern = "/"
    fig.xgrid.band_hatch_alpha = 0.6
    fig.xgrid.band_hatch_color = "lightgrey"
    fig.xgrid.band_hatch_weight = 0.5
    fig.xgrid.band_hatch_scale = 10

def flatten(l):
    return [item for sublist in l for item in sublist]

def main():
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    st.title("Featurized Persistent Barcode")
    
    tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
    
    menu = ["Cifar10","Fashion MNIST","Outex68", "Shrec14", "Custom"]
    choice = st.sidebar.selectbox("Select a Dataset",menu)

    mainPath = os.getcwd()

    filtrationType = ""

    if choice == "Cifar10":
        file_path = mainPath + r"/data/cifar10.png"
    elif choice == "Fashion MNIST":
        file_path = mainPath + r"/data/FashionMNIST.jpg"
        filtrationType = "Edge growing"
    elif choice == "Outex68":
        file_path = mainPath + r"/data/Outex1.bmp"
    elif choice == "Shrec14":
        file_path = mainPath + r"/data/shrec14_data_0.csv"
    else:
        file_path = st.sidebar.file_uploader("Upload Image",type=['png','jpeg','jpg','bmp','csv'])

    st.sidebar.markdown('#')
    st.sidebar.caption('''### About:''')
    st.sidebar.caption('''
        This web app can be used to compute and visualize featurized PH barcodes. This app is a part of the follwoing research:\\
        [Paper Refrence](https://google.com)\\
        This app can compute PH barcodes in dimension 0 and 1, and user can select corresponding check boxes to have them plotted 
        along with an option to plot persistent diagram. Furthermore, various barcode featurization methods can be selected to 
        compute and visualize as tables (e.g. Persistence statistics) or plots. Last but not least, an export button is associated 
        with each PH barcodes/diagrams and featurised PH barcodes so that users can easily download associated data to their 
        local machines.''')

    if file_path is not None:
        
        isShowImageChecked = st.checkbox('Input File', value=True)
        isPointCloud = (choice == "Shrec14" or 
                        (choice == "Custom" and os.path.splitext(file_path.name)[1] == ".csv"))

        if isPointCloud:
            input_data = load_csv(file_path)
        else:
            input_data = load_image(file_path)

        if isShowImageChecked:
            if isPointCloud:
                fig = px.scatter_3d(input_data, x='x', y='y', z='z')
                st.plotly_chart(fig)
            else:
                st.image(input_data,width=250)

        st.caption(f"Filtration type: {filtrationType}")
        
        if(choice == "Shrec14"):
            pd0 = pd.read_csv(mainPath + r"/data/shrec14_data_0_ph0.csv").to_numpy()
            pd1 = pd.read_csv(mainPath + r"/data/shrec14_data_0_ph1.csv").to_numpy()
        else:
            pd0, pd1 = GetPds(input_data, isPointCloud)

        isPersBarChecked = st.checkbox('Persistence Barcodes')

        if isPersBarChecked:
            st.subheader("Persistence Barcodes")
            source = ColumnDataSource(data={'left': pd0[:,0], 'right': pd0[:,1], 'y': range(len(pd0[:,0]))})
            fig = figure(title='Persistence Barcode [dim = 0]', height=250, tools = tools)
            fig.hbar(y='y', left ='left', right='right', height=0.1, alpha=0.7, source=source)
            st.bokeh_chart(fig, use_container_width=True)
            
            source = ColumnDataSource(data={'left': pd1[:,0], 'right': pd1[:,1], 'y': range(len(pd1[:,0]))})
            fig = figure(title='Persistence Barcode [dim = 1]', height=250, tools = tools)
            fig.hbar(y='y', left ='left', right='right', height=0.1, color="darkorange", alpha=0.7, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('PH0', pd0)
            CreateDownloadButton('PH1', pd1)
            st.markdown('#')

        isPersDiagChecked = st.checkbox('Persistence Diagram')

        if isPersDiagChecked:
            st.subheader("Persistence Diagram")
            fig, ax = plt.subplots()
            persim.plot_diagrams([pd0,pd1],labels= ["H0", "H1"], ax=ax)
            ax.set_title("Persistence Diagram")
            st.pyplot(fig)
        
        isBettiCurveChecked = st.checkbox('Betti Curve')

        if isBettiCurveChecked:
            tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
            st.subheader("Betti Curve")
            st.slider("Resolution", 0, 100, value=60, step=1, key='BettiCurveRes')

            Btt_0 = vec.GetBettiCurveFeature(pd0, st.session_state.BettiCurveRes)
            source = ColumnDataSource(data={'x': range(0, len(Btt_0)), 'y': Btt_0})
            fig = figure(x_range=(0, len(Btt_0)),title='Betti Curve [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            Btt_1 = vec.GetBettiCurveFeature(pd1, st.session_state.BettiCurveRes)
            source = ColumnDataSource(data={'x': range(0, len(Btt_1)), 'y': Btt_1})
            fig = figure(x_range=(0, len(Btt_1)), title='Betti Curve [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='red', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Betti Curve (PH0)', Btt_0)
            CreateDownloadButton('Betti Curve (PH1)', Btt_1)
            st.markdown('#')

        isPersStatsChecked = st.checkbox('Persistent Statistics')

        if isPersStatsChecked:
            st.subheader("Persistent Statistics")
            stat_0 = vec.GetPersStats(pd0)
            stat_1 = vec.GetPersStats(pd1)
            df = pd.DataFrame(np.array((stat_0[0:6], stat_1[0:6])), index=['PH(0)', 'PH(1)'])
            df.columns =['Birth Average', 'Death Average', 'Birth STD.', 'Death STD.', 
                         'Birth Median', 'Death Median']
            st.dataframe(df)

            df = pd.DataFrame(np.array((stat_0[6:11], stat_1[6:11])), index=['PH(0)', 'PH(1)'])
            df.columns =['Bar Length Average', 'Bar Length STD', 
                         'Bar Length Median', 'Bar Count', 'Persistent Entropy']
            st.dataframe(df)

            CreateDownloadButton('Persistent Stats (PH0)', stat_0)
            CreateDownloadButton('Persistent Stats (PH1)', stat_1)
            st.markdown('#')

        isPersImgChecked = st.checkbox('Persistence Image')

        if isPersImgChecked:
            st.subheader("Persistence Image")
            st.slider("Resolution", 0, 100, value=60, step=1, key='PersistenceImageRes')

            col1, col2 = st.columns(2)
            PI_0 = vec.GetPersImageFeature(pd0, st.session_state.PersistenceImageRes)
            fig, ax = plt.subplots()
            ax.imshow(np.flip(np.reshape(PI_0, [st.session_state.PersistenceImageRes,st.session_state.PersistenceImageRes]), 0))
            ax.set_title("Persistence Image [dim = 0]")
            col1.pyplot(fig)

            PI_1 = vec.GetPersImageFeature(pd1, st.session_state.PersistenceImageRes)
            fig, ax = plt.subplots()
            ax.imshow(np.flip(np.reshape(PI_1, [st.session_state.PersistenceImageRes,st.session_state.PersistenceImageRes]), 0))
            ax.set_title("Persistence Image [dim = 1]")
            col2.pyplot(fig)

            CreateDownloadButton('Persistence Image (PH0)', PI_0)
            CreateDownloadButton('Persistence Image (PH1)', PI_1)
            st.markdown('#')

        isPersLandChecked = st.checkbox('Persistence Landscapes')

        if isPersLandChecked:
            st.subheader("Persistence Landscapes")
            st.caption("Default parameters using persim package")
            # col1, col2 = st.columns(2)
            fig, ax = plt.subplots()
            fig.set_figheight(3)
            PL_0 = landscapes.visuals.PersLandscapeExact([pd0],hom_deg=0)
            landscapes.visuals.plot_landscape_simple(PL_0, ax=ax)
            ax.set_title("Persistence Landscapes [dim = 0]")
            ax.get_legend().remove()
            fig.tight_layout()
            st.pyplot(fig)

            fig, ax = plt.subplots()
            fig.set_figheight(3)
            PL_1 = landscapes.visuals.PersLandscapeExact([pd0, pd1],hom_deg=1)
            landscapes.visuals.plot_landscape_simple(PL_1, ax=ax)
            ax.set_title("Persistence Landscapes [dim = 1]")
            ax.get_legend().remove()
            fig.tight_layout()
            st.pyplot(fig)

            PL_0_csv = [flatten(i) for i in PL_0.compute_landscape()]
            CreateDownloadButton('Persistence Landscapes (PH0)', PL_0_csv)

            PL_1_csv = [flatten(i) for i in PL_1.compute_landscape()]
            CreateDownloadButton('Persistence Landscapes (PH1)', PL_1_csv)

            st.markdown('#')

        isPersEntropyChecked = st.checkbox('Entropy Summary Function')

        if isPersEntropyChecked:
            st.subheader("Entropy Summary Function")
            st.slider("Resolution", 0, 100, value=60, step=1, key='PersEntropyRes')

            PersEntropy_0 = vec.GetPersEntropyFeature(pd0, res=st.session_state.PersEntropyRes)
            source = ColumnDataSource(data={'x': range(0, len(PersEntropy_0)), 'y': PersEntropy_0})
            fig = figure(x_range=(0, len(PersEntropy_0)),title='Entropy Summary Function [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            PersEntropy_1 = vec.GetPersEntropyFeature(pd1, res=st.session_state.PersEntropyRes)
            source = ColumnDataSource(data={'x': range(0, len(PersEntropy_1)), 'y': PersEntropy_1})
            fig = figure(x_range=(0, len(PersEntropy_1)),title='Entropy Summary Function [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='red', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Entropy Summary Function (PH0)', PersEntropy_0)
            CreateDownloadButton('Entropy Summary Function (PH1)', PersEntropy_1)
            st.markdown('#')

        isPersSilChecked = st.checkbox('Persistence Silhouette')

        if isPersSilChecked:
            st.subheader("Persistence Silhouette")
            st.latex(r'''\textrm{$Weight$ $function$} = (q-p)^w''')
            st.latex(r'''\textrm{ where $w = 1$, $p$ and $q$ are called birth and death of a bar, respectively}''')
            st.slider("Resolution", 0, 100, value=60, step=1, key='PersSilRes')

            PersSil_0 = vec.GetPersSilhouetteFeature(pd0, res=st.session_state.PersSilRes)
            source = ColumnDataSource(data={'x': range(0, len(PersSil_0)), 'y': PersSil_0})
            fig = figure(x_range=(0, len(PersSil_0)),title='Persistence Silhouette [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            PersSil_1 = vec.GetPersSilhouetteFeature(pd1, res=st.session_state.PersSilRes)
            source = ColumnDataSource(data={'x': range(0, len(PersSil_1)), 'y': PersSil_1})
            fig = figure(x_range=(0, len(PersSil_1)),title='Persistence Silhouette [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='red', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Persistence Silhouette (PH0)', PersSil_0)
            CreateDownloadButton('Persistence Silhouette (PH1)', PersSil_1)
            st.markdown('#')

        isAtolChecked = st.checkbox('Atol')

        if isAtolChecked:
            st.subheader("Atol")
            st.slider("Number of Clusters", 2, 10, value=4, step=1, key='AtolNumberClusters')
            atol = vec.GetAtolFeature([pd0, pd1], k = st.session_state.AtolNumberClusters)

            cat = [f'{i}' for i in range(len(atol[0]))]
            fig = figure(x_range=cat, title='Atol [dim = 0]', height=250, tools = tools)
            fig.vbar(x=cat, top=atol[0], width=0.9, color="darkblue", alpha=0.5)
            fig.xaxis.axis_label = "Clusters"
            st.bokeh_chart(fig, use_container_width=True)

            cat = [f'{i}' for i in range(len(atol[1]))]
            fig = figure(x_range=cat, title='Atol [dim = 1]', height=250, tools = tools)
            fig.vbar(x=cat, top=atol[1], width=0.9, color="darkred", alpha=0.5)
            fig.xaxis.axis_label = "Clusters"
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Atol (PH0)', atol[0])
            CreateDownloadButton('Atol (PH1)', atol[1])
            st.markdown('#')

        isCarlsCoordsChecked = st.checkbox('Algebraic Coordinates')

        if isCarlsCoordsChecked:
            st.subheader("Algebraic Coordinates")
            st.caption("Number of features = 5")
            
            carlsCoords_0 = vec.GetCarlssonCoordinatesFeature(pd0)
            fig = figure(title='Algebraic Coordinates [dim = 0]', height=500, tools = tools)
            fig.vbar(x=range(len(carlsCoords_0)), top=carlsCoords_0, width=0.9, color="darkblue", alpha=0.5)
            fig.xaxis.major_label_overrides = {
                0: r"$$f_1 = \sum_i p_i(q_i - p_i)$$",
                1: r"$$f_2 =\sum_i(q_+-q_i)\,(q_i -p_i)$$",
                2: r"$$f_3 = \sum_i p_i^2(q_i - p_i)^4$$",
                3: r"$$f_4 = \sum_i(q_+-q_i)^2\,(q_i - p_i)^4$$",
                4: r"$$f_5 = \sum \text{max}(q_i-p_i)$$"
            }

            fig.xaxis.major_label_orientation = 0.8
            st.bokeh_chart(fig, use_container_width=True)

            carlsCoords_1 = vec.GetCarlssonCoordinatesFeature(pd1)
            fig = figure(title='Algebraic Coordinates [dim = 1]', height=500, tools = tools)
            fig.vbar(x=range(len(carlsCoords_1)), top=carlsCoords_1, width=0.9, color="darkred", alpha=0.5)
            fig.xaxis.major_label_overrides = {
                0: r"$$f_1 = \sum_i p_i(q_i - p_i)$$",
                1: r"$$f_2 =\sum_i(q_\text{max}-q_i)\,(q_i -p_i)$$",
                2: r"$$f_3 = \sum_i p_i^2(q_i - p_i)^4$$",
                3: r"$$f_4 = \sum_i(q_\text{max}-q_i)^2\,(q_i - p_i)^4$$",
                4: r"$$f_5 = \sum \text{max}(q_i-p_i)$$"
            }

            fig.xaxis.major_label_orientation = 0.8
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Algebraic Coordinates (PH0)', carlsCoords_0)
            CreateDownloadButton('Algebraic Coordinates (PH1)', carlsCoords_1)
            st.markdown('#')

        isPersLifeSpanChecked = st.checkbox('Lifespan Curve')

        if isPersLifeSpanChecked:
            st.subheader("Lifespan Curve")
            st.slider("Resolution", 0, 100, value=60, step=1, key='PersLifeSpanRes')

            persLifeSpan_0 = vec.GetPersLifespanFeature(pd0, res=st.session_state.PersLifeSpanRes)
            xrange = (0, len(persLifeSpan_0))
            source = ColumnDataSource(data={'x': range(0, len(persLifeSpan_0)), 'y': persLifeSpan_0})
            fig = figure(x_range=xrange, title='Lifespan Curve [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            persLifeSpan_1 = vec.GetPersLifespanFeature(pd1, res=st.session_state.PersLifeSpanRes)
            xrange = (0, len(persLifeSpan_1))
            source = ColumnDataSource(data={'x': range(0, len(persLifeSpan_1)), 'y': persLifeSpan_1})
            fig = figure(x_range=xrange, title='Lifespan Curve [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='red', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Lifespan Curve (PH0)', persLifeSpan_0)
            CreateDownloadButton('Lifespan Curve (PH1)', persLifeSpan_1)
            st.markdown('#')

        isComplexPolynomialChecked = st.checkbox('Complex Polynomial')

        if isComplexPolynomialChecked:
            st.subheader("Complex Polynomial")

            tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
            st.selectbox("Polynomial Type",["R", "S", "T"], index=0, key='CPType')

            CP_pd0 = vec.GetComplexPolynomialFeature(pd0, pol_type=st.session_state.CPType)
            source = ColumnDataSource(data={'x': CP_pd0[:,0], 'y': CP_pd0[:,1]})
            fig = figure(title='Complex Polynomial [dim = 0]', height=250, tools = tools)
            fig.circle(x='x', y='y', color="darkblue", alpha=0.4, size=5, hover_color="red", source=source)
            fig.xaxis.axis_label = "Real"
            fig.yaxis.axis_label = "Imaginary"
            st.bokeh_chart(fig, use_container_width=True)

            CP_pd1 = vec.GetComplexPolynomialFeature(pd1, pol_type=st.session_state.CPType)
            source = ColumnDataSource(data={'x': CP_pd1[:,0], 'y': CP_pd1[:,1]})
            fig = figure(title='Complex Polynomial [dim = 1]', height=250, tools = tools)
            fig.circle(x='x', y='y', color="darkred", alpha=0.4, size=5, hover_color="red", source=source)
            fig.xaxis.axis_label = "Real"
            fig.yaxis.axis_label = "Imaginary"
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Complex Polynomial (PH0)', CP_pd0)
            CreateDownloadButton('Complex Polynomial (PH1)', CP_pd1)
            st.markdown('#')

        isTopologicalVectorChecked = st.checkbox('Topological Vector')

        if isTopologicalVectorChecked:
            st.subheader("Topological Vector")
            st.slider("Threshold", 2, 10, value=8, step=1, key='TopologicalVectorThreshold')

            topologicalVector_0 = vec.GetTopologicalVectorFeature(pd0, thres=st.session_state.TopologicalVectorThreshold)
            cat = [f'{i}' for i in range(len(topologicalVector_0))]
            fig = figure(x_range=cat, title='Topological Vector [dim = 0]', height=250, tools = tools)
            fig.vbar(x=cat, top=topologicalVector_0, width=0.8, color="darkblue", alpha=0.5)
            fig.xaxis.axis_label = "Element"
            fig.yaxis.axis_label = "Threshold"
            st.bokeh_chart(fig, use_container_width=True)

            topologicalVector_1 = vec.GetTopologicalVectorFeature(pd1, thres=st.session_state.TopologicalVectorThreshold)
            cat = [f'{i}' for i in range(len(topologicalVector_0))]
            fig = figure(x_range=cat, title='Topological Vector [dim = 1]', height=250, tools = tools)
            fig.vbar(x=cat, top=topologicalVector_1, width=0.8, color="darkred", alpha=0.5)
            fig.xaxis.axis_label = "Element"
            fig.yaxis.axis_label = "Threshold"
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Topological Vector (PH0)', topologicalVector_0)
            CreateDownloadButton('Topological Vector (PH1)', topologicalVector_1)
            st.markdown('#')

        isPersTropCoordsChecked = st.checkbox('Tropical Coordinates')

        if isPersTropCoordsChecked:
            st.subheader("Tropical Coordinates")

            persTropCoords_0 = vec.GetPersTropicalCoordinatesFeature(pd0)
            fig = figure(title='Tropical Coordinates [dim = 0]', height=250, tools = tools)
            fig.vbar(x=range(len(persTropCoords_0)), top=persTropCoords_0, width=0.9, color="darkblue", alpha=0.5)
            fig.xaxis.axis_label = "Coordinate"
            fig.xaxis.major_label_overrides = {i: f'{i+1}' for i in range(len(persTropCoords_0))}
            st.bokeh_chart(fig, use_container_width=True)

            persTropCoords_1 = vec.GetPersTropicalCoordinatesFeature(pd1)
            fig = figure(title='Tropical Coordinates [dim = 1]', height=250, tools = tools)
            fig.vbar(x=range(len(persTropCoords_0)), top=persTropCoords_1, width=0.9, color="darkred", alpha=0.5)
            fig.xaxis.axis_label = "Coordinate"
            fig.xaxis.major_label_overrides = {i: f'{i+1}' for i in range(len(persTropCoords_1))}
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Tropical Coordinates (PH0)', persTropCoords_0)
            CreateDownloadButton('Tropical Coordinates (PH1)', persTropCoords_1)
            st.markdown('#')

        isTemplateFunctionChecked = st.checkbox('Template Function')

        if isTemplateFunctionChecked:
            st.subheader("Template Function")

        isTemplateSystemChecked = st.checkbox('Template System')

        if isTemplateSystemChecked:
            st.subheader("Template System")


if __name__ == '__main__':
    main()
