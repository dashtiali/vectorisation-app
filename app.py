import streamlit as st
from PIL import Image
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import vectorisation as vec
import plotly.express as px
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Range1d
from bokeh.transform import factor_cmap
import pandas as pd
import io
import os
from scipy.spatial import distance
from ripser import ripser


@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img
    
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

def PlotPersistantDiagram(pd0, pd1):
    if (len(pd0.shape) > 1):
        pd0 = pd0[pd0[:, 0] != 256]
        pd0 = pd0[pd0[:, 1] != 256]

    if (len(pd1.shape) > 1):
        pd1 = pd1[pd1[:, 0] != np.inf]
        pd1 = pd1[pd1[:, 1] != np.inf]

    tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]

    fig = figure(height=400, tools = tools)
    fig.xaxis.axis_label = 'Birth'
    fig.yaxis.axis_label = 'Death'

    col1 = np.append(np.full(len(pd0[:,0]), 'PH0'), np.full(len(pd1[:,0]), 'PH1'))
    col2 = np.append(pd0[:,0], pd1[:,0])
    col3 = np.append(pd0[:,1], pd1[:,1])
    data = pd.DataFrame(data={'dim': col1, 'x': col2, 'y': col3})

    index_cmap = factor_cmap('dim', palette=['red', 'blue'], factors=sorted(data.dim.unique()))
    fig.scatter("x", "y", source=data,
    legend_group="dim", fill_alpha=0.2, size=4,
    marker="circle",line_color=None,
    fill_color=index_cmap)

    fig.legend.location = "top_left"
    fig.legend.title = "Persistent Diagram"


    maxDim0 = np.max(pd0) if len(pd0) > 0 else 0
    maxDim1 = np.max(pd1) if len(pd1) > 0 else 0
    lineY = max(maxDim0, maxDim1) + 2
    fig.line([-1,lineY], [-1, lineY], line_width=1, color='black', alpha=0.5)
    fig.x_range = Range1d(-1, lineY)
    fig.y_range = Range1d(-1, lineY)
    # chart.set(xlim=(-1, lineY), ylim=(-1, lineY))
    return fig

def CreateDownloadButton(label, data):
    # Create an in-memory buffer
    with io.BytesIO() as buffer:
        # Write array to buffer
        np.savetxt(buffer, data, delimiter=",")
        st.download_button(
            label=f"Download {label}",
            data = buffer, # Download buffer
            file_name = f'{label}.csv',
            mime='text/csv')

def main():
    # hide_menu_style = """
    #     <style>
    #     #MainMenu {visibility: hidden;}
    #     footer {visibility: hidden;}
    #     </style>
    #     """
    # st.markdown(hide_menu_style, unsafe_allow_html=True)

    st.title("Featurized Persistent Barcode")
    
    tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
    
    menu = ["Cifar10","Fashion MNIST","Outex68", "Shrec14", "Custom"]
    choice = st.sidebar.selectbox("Select a Dataset",menu)

    mainPath = os.getcwd()

    if choice == "Cifar10":
        file_path = mainPath + r"/data/cifar10.png"
    elif choice == "Fashion MNIST":
        file_path = mainPath + r"/data/FashionMNIST.jpg"
    elif choice == "Outex68":
        file_path = mainPath + r"/data/Outex1.bmp"
    elif choice == "Shrec14":
        file_path = mainPath + r"/data/shrec14_data_0.csv"
    else:
        file_path = st.sidebar.file_uploader("Upload Image",type=['png','jpeg','jpg','bmp','csv'])

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
        
        if(choice == "Shrec14"):
            pd0 = pd.read_csv(r"data\shrec14_data_0_ph0.csv").to_numpy()
            pd1 = pd.read_csv(r"data\shrec14_data_0_ph1.csv").to_numpy()
        else:
            pd0, pd1 = GetPds(input_data, isPointCloud)

        isPersBarChecked = st.checkbox('Persistence Barcodes')

        if isPersBarChecked:
            st.subheader("Persistence Barcode")
            # col1, col2 = st.columns(2)
            # fig, ax = plt.subplots()
            # gd.plot_persistence_barcode(pd0, axes=ax)
            # ax.set_title("Persistence Barcode [dim = 0]")
            # col1.pyplot(fig)

            source = ColumnDataSource(data={'x1': pd0[:,0], 'x2': pd0[:,1] - pd0[:,0], 'y': range(len(pd0[:,0]))})
            fig = figure(title='Persistence Barcode [dim = 0]', height=250, tools = tools)
            fig.hbar(y='y', left='x1', right='x2', height=0.1, alpha=0.5, source=source)
            st.bokeh_chart(fig, use_container_width=True)
            
            source = ColumnDataSource(data={'x1': pd1[:,0], 'x2': pd1[:,1] - pd1[:,0], 'y': range(len(pd1[:,0]))})
            fig = figure(title='Persistence Barcode [dim = 1]', height=250, tools = tools)
            fig.hbar(y='y', left='x1', right='x2', height=0.1, alpha=0.5, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('PH0', pd0)
            CreateDownloadButton('PH1', pd1)
            st.markdown('#')

        isPersDiagChecked = st.checkbox('Persistence Diagram')

        if isPersDiagChecked:
            st.subheader("Persistence Diagram")
            # fig, ax = plt.subplots()
            # gd.plot_persistence_diagram(pds, axes=ax)
            # ax.set_title("Persistence Diagram")
            # st.pyplot(fig)
            fig = PlotPersistantDiagram(pd0, pd1)
            st.bokeh_chart(fig, use_container_width=True)
        
        isBettiCurveChecked = st.checkbox('Betti Curve')

        if isBettiCurveChecked:
            tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
            st.subheader("Betti Curve")

            st.slider("Resolution", 0, 100, value=60, step=1, key='BettiCurveRes')

            Btt_0 = vec.GetBettiCurveFeature(pd0, st.session_state.BettiCurveRes)
            source = ColumnDataSource(data={'x': range(0, len(Btt_0)), 'y': Btt_0})
            fig = figure(title='Betti Curve [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, source=source)
            fig.circle(x='x', y='y', fill_color="darkblue", alpha=0.4, size=4, hover_color="red", source=source)
            st.bokeh_chart(fig, use_container_width=True)

            Btt_1 = vec.GetBettiCurveFeature(pd1, st.session_state.BettiCurveRes)
            fig = figure(title='Betti Curve [dim = 1]', height=250, tools = tools)
            fig.line(range(0, len(Btt_1)), Btt_1, color='blue', alpha=0.5)
            fig.circle(range(0, len(Btt_1)), Btt_1, fill_color="darkblue", alpha=0.4, size=4, hover_color="red")
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Betti Curve (PH0)', Btt_0)
            CreateDownloadButton('Betti Curve (PH1)', Btt_1)
            st.markdown('#')

        isPersStatsChecked = st.checkbox('Persistent Statistics')

        if isPersStatsChecked:
            st.subheader("Persistent Statistics")
            stat_0 = vec.GetPersStats(pd0)
            stat_1 = vec.GetPersStats(pd1)
            df = pd.DataFrame(np.array((stat_0, stat_1)), index=['PH(0)', 'PH(1)'])
            df.columns =['Birth Average', 'Death Average', 'Birth STD.', 'Death STD.', 
                         'Birth Median', 'Death Median', 'Bar Length Average', 'Bar Length STD', 
                         'Bar Length Median', 'Bar Count', 'Barcode Persitent Entropy']
            st.dataframe(df)

            CreateDownloadButton('Persistent Stats (PH0)', stat_0)
            CreateDownloadButton('Persistent Stats (PH1)', stat_1)
            st.markdown('#')

        isPersImgChecked = st.checkbox('Persistent Image')

        if isPersImgChecked:
            st.subheader("Persistent Image")
            col1, col2 = st.columns(2)
            res = [100,100]
            PI_0 = vec.GetPersImageFeature(pd0, res)
            fig, ax = plt.subplots()
            ax.imshow(np.flip(np.reshape(PI_0, res), 0))
            ax.set_title("Persistent Image [dim = 0]")
            col1.pyplot(fig)
        
            PI_1 = vec.GetPersImageFeature(pd1, res)
            fig, ax = plt.subplots()
            ax.imshow(np.flip(np.reshape(PI_1, res), 0))
            ax.set_title("Persistent Image [dim = 1]")
            col2.pyplot(fig)

            CreateDownloadButton('Persistent Image (PH0)', PI_0)
            CreateDownloadButton('Persistent Image (PH1)', PI_1)
            st.markdown('#')

        isPersLandChecked = st.checkbox('Persistent Landscape')

        if isPersLandChecked:
            st.subheader("Persistent Landscape")
            col1, col2 = st.columns(2)
            PL_0 = vec.GetPersLandscapeFeature(pd0)
            fig, ax = plt.subplots()
            ax.plot(PL_0[:100])
            ax.plot(PL_0[100:200])
            ax.plot(PL_0[200:300])
            ax.set_title("Persistent Landscape [dim = 0]")
            col1.pyplot(fig)

            PL_1 = vec.GetPersLandscapeFeature(pd1)
            fig, ax = plt.subplots()
            ax.plot(PL_1[:100])
            ax.plot(PL_1[100:200])
            ax.plot(PL_1[200:300])
            ax.set_title("Persistent Landscape [dim = 1]")
            col2.pyplot(fig)

            CreateDownloadButton('Persistent Landscape (PH0)', PL_0)
            CreateDownloadButton('Persistent Landscape (PH1)', PL_1)
            st.markdown('#')

        isPersEntropyChecked = st.checkbox('Persistent Entropy')

        if isPersEntropyChecked:
            st.subheader("Persistent Entropy")
            PersEntropy_0 = vec.GetPersEntropyFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Persistent entropy [dim = 0]")
            st.line_chart(PersEntropy_0)

            PersEntropy_1 = vec.GetPersEntropyFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Persistent entropy [dim = 1]")
            st.line_chart(PersEntropy_1)

            CreateDownloadButton('Persistent Entropy (PH0)', PersEntropy_0)
            CreateDownloadButton('Persistent Entropy (PH1)', PersEntropy_1)
            st.markdown('#')

        isPersSilChecked = st.checkbox('Persistent Silhouette')

        if isPersSilChecked:
            st.subheader("Persistent silhouette")
            PersSil_0 = vec.GetPersSilhouetteFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Silhouette [dim = 0]")
            st.line_chart(PersSil_0)

            PersSil_1 = vec.GetPersSilhouetteFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Silhouette [dim = 1]")
            st.line_chart(PersSil_1)

            CreateDownloadButton('Persistent Silhouette (PH0)', PersSil_0)
            CreateDownloadButton('Persistent Silhouette (PH1)', PersSil_1)
            st.markdown('#')

        isAtolChecked = st.checkbox('Atol')

        if isAtolChecked:
            st.subheader("Atol")
            atol_0 = vec.GetAtolFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Atol [dim = 0]")
            st.line_chart(atol_0)

            atol_1 = vec.GetAtolFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Atol [dim = 1]")
            st.line_chart(atol_1)

            CreateDownloadButton('Atol (PH0)', atol_0)
            CreateDownloadButton('Atol (PH1)', atol_1)
            st.markdown('#')

        isCarlsCoordsChecked = st.checkbox('Carlsson Coordinates')

        if isCarlsCoordsChecked:
            st.subheader("Carlsson Coordinates")
            carlsCoords_0 = vec.GetCarlssonCoordinatesFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Carlsson Coordinates [dim = 0]")
            st.line_chart(carlsCoords_0)

            carlsCoords_1 = vec.GetCarlssonCoordinatesFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Carlsson Coordinates [dim = 1]")
            st.line_chart(carlsCoords_1)

            CreateDownloadButton('Carlsson Coordinates (PH0)', carlsCoords_0)
            CreateDownloadButton('Carlsson Coordinates (PH1)', carlsCoords_1)
            st.markdown('#')

        isPersLifeSpanChecked = st.checkbox('Persistent Life Span')

        if isPersLifeSpanChecked:
            st.subheader("Persistent Life Span")
            persLifeSpan_0 = vec.GetPersLifespanFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Life Span [dim = 0]")
            st.line_chart(persLifeSpan_0)

            persLifeSpan_1 = vec.GetPersLifespanFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Life Span [dim = 1]")
            st.line_chart(persLifeSpan_1)

            CreateDownloadButton('Persistent Life Span (PH0)', persLifeSpan_0)
            CreateDownloadButton('Persistent Life Span (PH1)', persLifeSpan_1)
            st.markdown('#')

        isComplexPolynomialChecked = st.checkbox('Complex Polynomial')

        if isComplexPolynomialChecked:
            st.subheader("Complex Polynomial")

            tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
            st.selectbox("Polynomial Type",["R", "S", "T"], index=0, key='CPType')

            CP_pd0 = vec.GetComplexPolynomialFeature(pd0, pol_type=st.session_state.CPType)
            source = ColumnDataSource(data={'x': range(0, len(CP_pd0)), 'y': CP_pd0})
            fig = figure(title='Complex Polynomial [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, source=source)
            fig.circle(x='x', y='y', fill_color="darkblue", alpha=0.4, size=4, hover_color="red", source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CP_pd1 = vec.GetComplexPolynomialFeature(pd1, pol_type=st.session_state.CPType)
            source = ColumnDataSource(data={'x': range(0, len(CP_pd1)), 'y': CP_pd1})
            fig = figure(title='Complex Polynomial [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, source=source)
            fig.circle(x='x', y='y', fill_color="darkblue", alpha=0.4, size=4, hover_color="red", source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Complex Polynomial (PH0)', CP_pd0)
            CreateDownloadButton('Complex Polynomial (PH1)', CP_pd1)
            st.markdown('#')

        isTopologicalVectorChecked = st.checkbox('Topological Vector')

        if isTopologicalVectorChecked:
            st.subheader("Persistent Topological Vector")
            topologicalVector_0 = vec.GetTopologicalVectorFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Topological Vector [dim = 0]")
            st.line_chart(topologicalVector_0)

            topologicalVector_1 = vec.GetTopologicalVectorFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Topological Vector [dim = 1]")
            st.line_chart(topologicalVector_1)

            CreateDownloadButton('Persistent Topological Vector (PH0)', topologicalVector_0)
            CreateDownloadButton('Persistent Topological Vector (PH1)', topologicalVector_1)
            st.markdown('#')

        isPersTropCoordsChecked = st.checkbox('Persistent Tropical Coordinates')

        if isPersTropCoordsChecked:
            st.subheader("Persistent Tropical Coordinates")
            persTropCoords_0 = vec.GetPersTropicalCoordinatesFeature(pd0)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Tropical Coordinates [dim = 0]")
            st.line_chart(persTropCoords_0)

            persTropCoords_1 = vec.GetPersTropicalCoordinatesFeature(pd1)
            fig, ax = plt.subplots()
            ax.set_title("Persistent Tropical Coordinates [dim = 1]")
            st.line_chart(persTropCoords_1)

            CreateDownloadButton('Persistent Tropical Coordinates (PH0)', persTropCoords_0)
            CreateDownloadButton('Persistent Tropical Coordinates (PH1)', persTropCoords_1)
            st.markdown('#')


if __name__ == '__main__':
    main()
