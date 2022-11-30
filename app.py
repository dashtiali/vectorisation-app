'''
    Name: Vectorization Web App
    This app is a part of the follwoing paper:

    "A Survey of Vectorization Methods in Topological Data Analysis, 
    Dashti A Ali, Aras Asaad, Maria Jose Jimenez, Vidit Nanda, Eduardo Paluzo-Hidalgo,
    and Manuel Soriano-Trigueros"

    Licensed: 
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''

import streamlit as st
from PIL import Image
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import vectorisation as vec
import plotly.express as px
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import pandas as pd
import io
import os
from scipy.spatial import distance
from ripser import ripser
import persim
import landscapes as landscapes
import logging
import traceback
import datetime


@st.cache
def load_image(file_path):
    '''
	Load image file and convert to grayscale
	:param file_path: full path of the image file
	:type file_path: string
	:return: numpy 2D array -- Grayscale version of the loaded image
	'''
    img = Image.open(file_path)
    gray_img = img.convert("L")
    return gray_img


@st.cache
def load_point_cloud(file_path):
    '''
	Load point cloud from csv file
	:param file_path: full path of the csv file
	:type file_path: string
    :csv format: 3 columns without header
	:return: numpy 2D array -- A Nx3 pandas dataframe with ['x', 'y', 'z'] headers
	'''
    df = pd.read_csv(file_path, names=['x', 'y', 'z'])
    return df


@st.cache
def load_csv(file_path):
    '''
	Load csv file
	:param file_path: full path of the image file
	:type file_path: string
	:return: numpy 2D array
	'''
    arr = pd.read_csv(file_path).to_numpy()
    return arr


def infty_proj(x):
     '''
	Replace infinity values with 256 in a nupy array
	:param file_path: full path of the image file
	:type x: numpy array
	:return: numpy array
	'''
     return (256 if ~np.isfinite(x) else x)


@st.cache
def GetPds(data, isPointCloud):
    '''
	Compute persistence barcodes (H0, H1) from image or point cloud
	:param data: image or point cloud data
	:type data: numpy 2D array
    :param isPointCloud: indicates if the input data is a point cloud
	:type isPointCloud: bolean
	:return pd0, pd1:  numpy 2D arrays -- Persistence barcodes in dimention 0, 1
	'''
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
    '''
	Create strreamlit buttons to download a numpy array as csv
	:param label: name of the downloadable csv file
	:type label: string
    :param data: the numpy array to be downloaded
	:type data: numpy 2D array
	'''
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
    '''
	Apply hatch pattern on chart area of a bokeh figure
	:param fig: the bokeh figure to be changed
	:type fig: bokeh figure
	'''
    fig.ygrid.grid_line_color = None
    fig.xgrid.band_hatch_pattern = "/"
    fig.xgrid.band_hatch_alpha = 0.6
    fig.xgrid.band_hatch_color = "lightgrey"
    fig.xgrid.band_hatch_weight = 0.5
    fig.xgrid.band_hatch_scale = 10

    
def flatten(lst):
    '''
	Flatten a list of lists into one list
	:param lst: the list to be flatten
	:type lst: list
	:return: 1D list -- The flatten list
	'''
    return [item for sublist in lst for item in sublist]


def main():
    '''
	The main function to create the streamlit app
	'''

    # ************************************************
    #                   CSS Styles
    # ************************************************
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    
    # Populating page info and markups 
    st.set_page_config(page_title="Vectorization")
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.title("Persistent Barcodes Vectorization")
    st.info('To compute and visualize featurized PH barcodes')

    # Populating sidebar and its main menu
    sidebar_menu = ["Cifar10","Fashion MNIST","Outex68", "Shrec14", "Custom"]
    choice = st.sidebar.selectbox("Select a Dataset", sidebar_menu)
    filtrationType = "Cubical Complex"

    # Set main and train files path
    mainPath = os.getcwd()
    train_folder = mainPath + r"/data/training_pds/"

    # Initialize or reset train file paths
    train_pd0_file_paths = []
    train_pd1_file_paths = []
    train_file_paths = []

    # Populate sample file and the train files paths
    # for selected option from the sidebar main menu
    if choice == "Cifar10":
        file_path = mainPath + r"/data/cifar10.png"
        train_pd0_file_paths = [train_folder + "cifar30_ph0.csv", train_folder + "cifar51_ph0.csv"]
        train_pd1_file_paths = [train_folder + "cifar30_ph1.csv", train_folder + "cifar51_ph1.csv"]
    elif choice == "Fashion MNIST":
        file_path = mainPath + r"/data/FashionMNIST.jpg"
        train_pd0_file_paths = [train_folder + "FashionMNIST21_ph0.csv", train_folder + "FashionMNIST502_ph0.csv"]
        train_pd1_file_paths = [train_folder + "FashionMNIST21_ph1.csv", train_folder + "FashionMNIST502_ph1.csv"]
        filtrationType = "Edge growing"
    elif choice == "Outex68":
        file_path = mainPath + r"/data/Outex1.bmp"
        train_pd0_file_paths = [train_folder + "Outex2_ph0.csv", train_folder + "Outex3_ph0.csv"]
        train_pd1_file_paths = [train_folder + "Outex2_ph1.csv", train_folder + "Outex3_ph1.csv"]
    elif choice == "Shrec14":
        file_path = mainPath + r"/data/shrec14_data_0.csv"
        train_pd0_file_paths = [train_folder + "shrec14_data_1_ph0.csv", train_folder + "shrec14_data_2_ph0.csv"]
        train_pd1_file_paths = [train_folder + "shrec14_data_1_ph1.csv", train_folder + "shrec14_data_2_ph1.csv"]
        filtrationType = "Heat Kernel Signature"
    else:
        file_path = st.sidebar.file_uploader("Upload Image Or Point Cloud",type=['png','jpeg','jpg','bmp','csv'])
        if file_path is not None:
            selectedType = os.path.splitext(file_path.name)[1]
            train_file_paths = st.sidebar.file_uploader('''In order to compute Atol or Adaptive Template System features, you need to select 
                                                            some data  of the same type as the selected one.''',
                                                            type=[selectedType], accept_multiple_files=True)
    
    # ************************************************
    #              About Section (sidebar)
    # ************************************************
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

    # When file path is selected or changed the following wil be populated
    if file_path is not None:
        
        # ************************************************
        #        Load and visualize the input file
        # ************************************************
        isShowImageChecked = st.checkbox('Input File', value=True)
        isPointCloud = (choice == "Shrec14" or 
                        (choice == "Custom" and os.path.splitext(file_path.name)[1] == ".csv"))

        if (choice == "Custom" and os.path.splitext(file_path.name)[1] == ".csv"):
            filtrationType = "Rips"
        
        if isPointCloud:
            input_data = load_point_cloud(file_path)
        else:
            input_data = load_image(file_path)

        if isShowImageChecked:
            if isPointCloud:
                fig = px.scatter_3d(input_data, x='x', y='y', z='z', color='z')
                minValue = np.floor(input_data.to_numpy().min())
                maxValue = np.ceil(input_data.to_numpy().max())
                axisRange = [minValue, maxValue]
                fig.update_layout(
                scene = dict(xaxis = dict(nticks=5, range=axisRange,),
                             yaxis = dict(nticks=5, range=axisRange,),
                             zaxis = dict(nticks=5, range=axisRange,),),
                             margin=dict(r=0, l=0, b=0, t=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.image(input_data,width=250)
        
        st.caption(f"Filtration type: {filtrationType}")
        
        # ************************************************
        # 1. Compute persistence barcodes of the input file 
        # 2. Load pre-computed persistence barcodes of 
        # the train data or Load selected train files
        # and compute their persistence barcodes
        # ************************************************
        train_pd0s = list()
        train_pd1s = list()
        train_pd0s.clear()
        train_pd1s.clear()

        if(choice == "Shrec14"):
            pd0 = load_csv(mainPath + r"/data/shrec14_data_0_ph0.csv")
            pd1 = load_csv(mainPath + r"/data/shrec14_data_0_ph1.csv")
        else:
            pd0, pd1 = GetPds(input_data, isPointCloud)

        if (choice == "Custom"):
            if train_file_paths is not None:
                for file in train_file_paths:
                    if isPointCloud:
                        file_data = load_point_cloud(file)
                    else:
                        file_data = load_image(file)

                    file_pd0, file_pd1 = GetPds(file_data, isPointCloud)

                    if len(file_pd0) != 0:
                        train_pd0s.append(file_pd0)
                    if len(file_pd1) != 0:
                        train_pd1s.append(file_pd1)
        else:
            train_pd0s = [load_csv(p) for p in train_pd0_file_paths]
            train_pd1s = [load_csv(p) for p in train_pd1_file_paths]
        
        # Populate radio buttons for visualize all option
        visualizeAll = False
        visualizationMode = st.radio("Visualization Mode: ", ('Custom Selection', 'Select All'), horizontal=True)
        visualizeAll = visualizationMode == 'Select All'
        st.markdown("""---""")

        # Initialize chart tools for bokeh chart type
        tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]

        # ************************************************
        #         Computing Persistence Barcodes
        # ************************************************
        isPersBarChecked = False if visualizeAll else st.checkbox('Persistence Barcodes')

        if isPersBarChecked or visualizeAll:
            st.subheader("Persistence Barcodes")
            
            if len(pd0) != 0:
                source = ColumnDataSource(data={'left': pd0[:,0], 'right': pd0[:,1], 'y': range(len(pd0[:,0]))})
            else:
                source = ColumnDataSource(data={'left': [], 'right': [], 'y': []})
            fig = figure(title='Persistence Barcode [dim = 0]', height=250, tools = tools)
            fig.hbar(y='y', left ='left', right='right', height=0.1, alpha=0.7, source=source)
            st.bokeh_chart(fig, use_container_width=True)
            
            if len(pd1) != 0:
                source = ColumnDataSource(data={'left': pd1[:,0], 'right': pd1[:,1], 'y': range(len(pd1[:,0]))})
            else:
                source = ColumnDataSource(data={'left': [], 'right': [], 'y': []})
            fig = figure(title='Persistence Barcode [dim = 1]', height=250, tools = tools)
            fig.hbar(y='y', left ='left', right='right', height=0.1, color="darkorange", alpha=0.7, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('PH0', pd0)
            CreateDownloadButton('PH1', pd1)
            st.markdown('#')

        # ************************************************
        #         Plotting Persistence Diagram
        # ************************************************
        isPersDiagChecked = False if visualizeAll else st.checkbox('Persistence Diagram')

        if isPersDiagChecked or visualizeAll:
            st.subheader("Persistence Diagram")
            fig, ax = plt.subplots()
            dgms = []

            if len(pd0) != 0:
                dgms.append(pd0)
            
            if len(pd1) != 0:
                dgms.append(pd1)

            if len(dgms) != 0:
                persim.plot_diagrams(dgms,labels= ["H0", "H1"], ax=ax)
            
            ax.set_title("Persistence Diagram")
            st.pyplot(fig)
        
        # ************************************************
        #         Computing Betti Curve
        # ************************************************
        isBettiCurveChecked = False if visualizeAll else st.checkbox('Betti Curve')

        if isBettiCurveChecked or visualizeAll:
            tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
            st.subheader("Betti Curve")
            st.slider("Resolution", 0, 100, value=60, step=1, key='BettiCurveRes')

            if len(pd0) != 0:
                Btt_0 = vec.GetBettiCurveFeature(pd0, st.session_state.BettiCurveRes)
                source = ColumnDataSource(data={'x': range(0, len(Btt_0)), 'y': Btt_0})
            else:
                Btt_0 = []
                source = ColumnDataSource(data={'x': [], 'y': Btt_0})
            
            fig = figure(x_range=(0, len(Btt_0)),title='Betti Curve [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                Btt_1 = vec.GetBettiCurveFeature(pd1, st.session_state.BettiCurveRes)
                source = ColumnDataSource(data={'x': range(0, len(Btt_1)), 'y': Btt_1})
            else:
                Btt_1 = []
                source = ColumnDataSource(data={'x': [], 'y': Btt_1})
            
            fig = figure(x_range=(0, len(Btt_1)), title='Betti Curve [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='red', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Betti Curve (PH0)', Btt_0)
            CreateDownloadButton('Betti Curve (PH1)', Btt_1)
            st.markdown('#')

        # ************************************************
        #         Computing Persistent Statistics
        # ************************************************
        isPersStatsChecked = False if visualizeAll else st.checkbox('Persistent Statistics')

        if isPersStatsChecked or visualizeAll:
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

        # ************************************************
        #         Computing Persistence Image
        # ************************************************
        isPersImgChecked = False if visualizeAll else st.checkbox('Persistence Image')

        if isPersImgChecked or visualizeAll:
            st.subheader("Persistence Image")
            st.slider("Resolution", 0, 100, value=60, step=1, key='PersistenceImageRes')

            col1, col2 = st.columns(2)
            fig, ax = plt.subplots()

            if len(pd0) != 0:
                PI_0 = vec.GetPersImageFeature(pd0, st.session_state.PersistenceImageRes)
                ax.imshow(np.flip(np.reshape(PI_0, [st.session_state.PersistenceImageRes,st.session_state.PersistenceImageRes]), 0))
            else:
                PI_0 = []

            ax.set_title("Persistence Image [dim = 0]")
            col1.pyplot(fig)

            fig, ax = plt.subplots()

            if len(pd1) != 0:
                PI_1 = vec.GetPersImageFeature(pd1, st.session_state.PersistenceImageRes)
                ax.imshow(np.flip(np.reshape(PI_1, [st.session_state.PersistenceImageRes,st.session_state.PersistenceImageRes]), 0))
            else:
                PI_1 = []
                
            ax.set_title("Persistence Image [dim = 1]")
            col2.pyplot(fig)

            CreateDownloadButton('Persistence Image (PH0)', PI_0)
            CreateDownloadButton('Persistence Image (PH1)', PI_1)
            st.markdown('#')
        
        # ************************************************
        #         Computing Persistence Landscapes
        # ************************************************
        isPersLandChecked = False if visualizeAll else st.checkbox('Persistence Landscapes')

        if isPersLandChecked or visualizeAll:
            st.subheader("Persistence Landscapes")
            st.caption("Default parameters using persim package")
            # col1, col2 = st.columns(2)
            fig, ax = plt.subplots()
            fig.set_figheight(3)

            if len(pd0) != 0:
                PL_0 = landscapes.visuals.PersLandscapeExact([pd0],hom_deg=0)
                landscapes.visuals.plot_landscape_simple(PL_0, ax=ax)
                ax.get_legend().remove()
                PL_0_csv = [flatten(i) for i in PL_0.compute_landscape()]
            else:
                PL_0 = []
                PL_0_csv = []
            
            ax.set_title("Persistence Landscapes [dim = 0]")
            fig.tight_layout()
            st.pyplot(fig)

            fig, ax = plt.subplots()
            fig.set_figheight(3)

            if len(pd1) != 0:
                PL_1 = landscapes.visuals.PersLandscapeExact([pd0, pd1],hom_deg=1)
                landscapes.visuals.plot_landscape_simple(PL_1, ax=ax)
                ax.get_legend().remove()
                PL_1_csv = [flatten(i) for i in PL_1.compute_landscape()]
            else:
                PL_1 = []
                PL_1_csv = []
            
            ax.set_title("Persistence Landscapes [dim = 1]")
            fig.tight_layout()
            st.pyplot(fig)

            CreateDownloadButton('Persistence Landscapes (PH0)', PL_0_csv)
            CreateDownloadButton('Persistence Landscapes (PH1)', PL_1_csv)

            st.markdown('#')

        # ************************************************
        #       Computing Entropy Summary Function
        # ************************************************
        isPersEntropyChecked = False if visualizeAll else st.checkbox('Entropy Summary Function')

        if isPersEntropyChecked or visualizeAll:
            st.subheader("Entropy Summary Function")
            st.slider("Resolution", 0, 100, value=60, step=1, key='PersEntropyRes')
            
            if len(pd0) != 0:
                PersEntropy_0 = vec.GetPersEntropyFeature(pd0, res=st.session_state.PersEntropyRes)
                source = ColumnDataSource(data={'x': range(0, len(PersEntropy_0)), 'y': PersEntropy_0})
            else:
                PersEntropy_0 = []
                source = ColumnDataSource(data={'x': [], 'y': []})
            
            fig = figure(x_range=(0, len(PersEntropy_0)),title='Entropy Summary Function [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                PersEntropy_1 = vec.GetPersEntropyFeature(pd1, res=st.session_state.PersEntropyRes)
                source = ColumnDataSource(data={'x': range(0, len(PersEntropy_1)), 'y': PersEntropy_1})
            else:
                PersEntropy_1 = []
                source = ColumnDataSource(data={'x': [], 'y': []})
            
            fig = figure(x_range=(0, len(PersEntropy_1)),title='Entropy Summary Function [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='red', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Entropy Summary Function (PH0)', PersEntropy_0)
            CreateDownloadButton('Entropy Summary Function (PH1)', PersEntropy_1)
            st.markdown('#')

        # ************************************************
        #         Computing Persistence Silhouette
        # ************************************************
        isPersSilChecked = False if visualizeAll else st.checkbox('Persistence Silhouette')

        if isPersSilChecked or visualizeAll:
            st.subheader("Persistence Silhouette")
            st.latex(r'''\textrm{$Weight$ $function$} = (q-p)^w''')
            st.latex(r'''\textrm{ where $w = 1$, $p$ and $q$ are called birth and death of a bar, respectively}''')
            st.slider("Resolution", 0, 100, value=60, step=1, key='PersSilRes')

            if len(pd0) != 0:
                PersSil_0 = vec.GetPersSilhouetteFeature(pd0, res=st.session_state.PersSilRes)
                source = ColumnDataSource(data={'x': range(0, len(PersSil_0)), 'y': PersSil_0})
            else:
                PersSil_0 = []
                source = ColumnDataSource(data={'x': [], 'y': []})
            
            fig = figure(x_range=(0, len(PersSil_0)),title='Persistence Silhouette [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                PersSil_1 = vec.GetPersSilhouetteFeature(pd1, res=st.session_state.PersSilRes)
                source = ColumnDataSource(data={'x': range(0, len(PersSil_1)), 'y': PersSil_1})
            else:
                PersSil_1 = []
                source = ColumnDataSource(data={'x': [], 'y': []})
            
            fig = figure(x_range=(0, len(PersSil_1)),title='Persistence Silhouette [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='red', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Persistence Silhouette (PH0)', PersSil_0)
            CreateDownloadButton('Persistence Silhouette (PH1)', PersSil_1)
            st.markdown('#')

        # ************************************************
        #               Computing Atol
        # ************************************************
        isAtolChecked = False if visualizeAll else st.checkbox('Atol')

        if isAtolChecked or visualizeAll:
            st.subheader("Atol")

            if len(train_pd0s) > 0 and len(train_pd1s) > 0:
                Atol_train_pd0s = train_pd0s
                Atol_train_pd1s = train_pd1s

                st.slider("Number of Clusters", 2, 10, value=4, step=1, key='AtolNumberClusters')

                if len(pd0) != 0:
                    Atol_train_pd0s.insert(0, pd0)
                    Atol_train_pd1s.insert(0, pd1)

                    atol_0 = vec.GetAtolFeature(Atol_train_pd0s, k = st.session_state.AtolNumberClusters)
                    cat = [f'{i}' for i in range(len(atol_0[0]))]
                else:
                    atol_0 = []
                    cat = []
                
                fig = figure(x_range=cat, title='Atol [dim = 0]', height=250, tools = tools)
                
                if len(atol_0) != 0:
                    fig.vbar(x=cat, top=atol_0[0], width=0.9, color="darkblue", alpha=0.5)
                
                fig.xaxis.axis_label = "Clusters"
                st.bokeh_chart(fig, use_container_width=True)

                if len(pd1) != 0:
                    atol_1 = vec.GetAtolFeature(Atol_train_pd1s, k = st.session_state.AtolNumberClusters)
                    cat = [f'{i}' for i in range(len(atol_1[0]))]
                else:
                    atol_1 = []
                    cat = []
                
                fig = figure(x_range=cat, title='Atol [dim = 1]', height=250, tools = tools)

                if len(atol_1) != 0:
                    fig.vbar(x=cat, top=atol_1[0], width=0.9, color="darkred", alpha=0.5)

                fig.xaxis.axis_label = "Clusters"
                st.bokeh_chart(fig, use_container_width=True)

                CreateDownloadButton('Atol (PH0)', atol_0[0])
                CreateDownloadButton('Atol (PH1)', atol_1[0])
            else:
                st.error('''In order to compute Atol features, you need to select at least 
                            one more data of the same type as the selected one. Use the second uploader from the sidebar on the left. 
                            If you have already done this step, so selected data did not produce any barcode, select another file.''')
            
            st.markdown('#')

        # ************************************************
        #         Computing Algebraic Coordinates
        # ************************************************
        isCarlsCoordsChecked = False if visualizeAll else st.checkbox('Algebraic Coordinates')

        if isCarlsCoordsChecked or visualizeAll:
            st.subheader("Algebraic Coordinates")
            st.caption("Number of features = 5")
            
            fig = figure(title='Algebraic Coordinates [dim = 0]', height=500, tools = tools)

            if len(pd0) != 0:
                carlsCoords_0 = vec.GetCarlssonCoordinatesFeature(pd0)
                fig.vbar(x=range(len(carlsCoords_0)), top=carlsCoords_0, width=0.9, color="darkblue", alpha=0.5)
            else:
                carlsCoords_0 = []
            
            fig.xaxis.major_label_overrides = {
                0: r"$$f_1 = \sum_i p_i(q_i - p_i)$$",
                1: r"$$f_2 =\sum_i(q_+-q_i)\,(q_i -p_i)$$",
                2: r"$$f_3 = \sum_i p_i^2(q_i - p_i)^4$$",
                3: r"$$f_4 = \sum_i(q_+-q_i)^2\,(q_i - p_i)^4$$",
                4: r"$$f_5 = \max_i\{(q_i-p_i)\}$$"
            }

            fig.xaxis.major_label_orientation = 0.8
            st.bokeh_chart(fig, use_container_width=True)

            fig = figure(title='Algebraic Coordinates [dim = 1]', height=500, tools = tools)

            if len(pd1) != 0:
                carlsCoords_1 = vec.GetCarlssonCoordinatesFeature(pd1)
                fig.vbar(x=range(len(carlsCoords_1)), top=carlsCoords_1, width=0.9, color="darkred", alpha=0.5)
            else:
                carlsCoords_1 = []
            
            fig.xaxis.major_label_overrides = {
                0: r"$$f_1 = \sum_i p_i(q_i - p_i)$$",
                1: r"$$f_2 =\sum_i(q_\text{max}-q_i)\,(q_i -p_i)$$",
                2: r"$$f_3 = \sum_i p_i^2(q_i - p_i)^4$$",
                3: r"$$f_4 = \sum_i(q_\text{max}-q_i)^2\,(q_i - p_i)^4$$",
                4: r"$$f_5 = \max_i\{(q_i-p_i)\}$$"
            }

            fig.xaxis.major_label_orientation = 0.8
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Algebraic Coordinates (PH0)', carlsCoords_0)
            CreateDownloadButton('Algebraic Coordinates (PH1)', carlsCoords_1)
            st.markdown('#')

        # ************************************************
        #         Computing Lifespan Curve
        # ************************************************
        isPersLifeSpanChecked = False if visualizeAll else st.checkbox('Lifespan Curve')

        if isPersLifeSpanChecked or visualizeAll:
            st.subheader("Lifespan Curve")
            st.slider("Resolution", 0, 100, value=60, step=1, key='PersLifeSpanRes')
            
            if len(pd0) != 0:
                persLifeSpan_0 = vec.GetPersLifespanFeature(pd0, res=st.session_state.PersLifeSpanRes)
                xrange = (0, len(persLifeSpan_0))
                source = ColumnDataSource(data={'x': range(0, len(persLifeSpan_0)), 'y': persLifeSpan_0})
            else:
                persLifeSpan_0 = []
                xrange = []
                source = ColumnDataSource(data={'x': [], 'y': []})
            
            fig = figure(x_range=xrange, title='Lifespan Curve [dim = 0]', height=250, tools = tools)
            fig.line(x='x', y='y', color='blue', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                persLifeSpan_1 = vec.GetPersLifespanFeature(pd1, res=st.session_state.PersLifeSpanRes)
                xrange = (0, len(persLifeSpan_1))
                source = ColumnDataSource(data={'x': range(0, len(persLifeSpan_1)), 'y': persLifeSpan_1})
            else:
                persLifeSpan_1 = []
                xrange = []
                source = ColumnDataSource(data={'x': [], 'y': []})
            
            fig = figure(x_range=xrange, title='Lifespan Curve [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='red', alpha=0.5, line_width=2, source=source)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Lifespan Curve (PH0)', persLifeSpan_0)
            CreateDownloadButton('Lifespan Curve (PH1)', persLifeSpan_1)
            st.markdown('#')

        # ************************************************
        #         Computing Complex Polynomial
        # ************************************************
        isComplexPolynomialChecked = False if visualizeAll else st.checkbox('Complex Polynomial')

        if isComplexPolynomialChecked or visualizeAll:
            st.subheader("Complex Polynomial")

            tools = ["pan","box_zoom", "wheel_zoom", "box_select", "hover", "reset"]
            st.selectbox("Polynomial Type",["R", "S", "T"], index=0, key='CPType')

            if len(pd0) != 0:
                CP_pd0 = vec.GetComplexPolynomialFeature(pd0, pol_type=st.session_state.CPType)
                source = ColumnDataSource(data={'x': CP_pd0[:,0], 'y': CP_pd0[:,1]})
            else:
                CP_pd0 = []
                source = ColumnDataSource(data={'x': [], 'y': []})
            
            fig = figure(title='Complex Polynomial [dim = 0]', height=250, tools = tools)
            fig.circle(x='x', y='y', color="darkblue", alpha=0.4, size=5, hover_color="red", source=source)
            fig.xaxis.axis_label = "Real"
            fig.yaxis.axis_label = "Imaginary"
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                CP_pd1 = vec.GetComplexPolynomialFeature(pd1, pol_type=st.session_state.CPType)
                source = ColumnDataSource(data={'x': CP_pd1[:,0], 'y': CP_pd1[:,1]})
            else:
                CP_pd1 = []
                source = ColumnDataSource(data={'x': [], 'y': []})
            
            fig = figure(title='Complex Polynomial [dim = 1]', height=250, tools = tools)
            fig.circle(x='x', y='y', color="darkred", alpha=0.4, size=5, hover_color="red", source=source)
            fig.xaxis.axis_label = "Real"
            fig.yaxis.axis_label = "Imaginary"
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Complex Polynomial (PH0)', CP_pd0)
            CreateDownloadButton('Complex Polynomial (PH1)', CP_pd1)
            st.markdown('#')

        # ************************************************
        #         Computing Topological Vector
        # ************************************************
        isTopologicalVectorChecked = False if visualizeAll else st.checkbox('Topological Vector')

        if isTopologicalVectorChecked or visualizeAll:
            st.subheader("Topological Vector")
            st.slider("Threshold", 2, 10, value=8, step=1, key='TopologicalVectorThreshold')

            if len(pd0) != 0:
                topologicalVector_0 = vec.GetTopologicalVectorFeature(pd0, thres=st.session_state.TopologicalVectorThreshold)
                cat = [f'{i}' for i in range(len(topologicalVector_0))]
            else:
                topologicalVector_0 = []
                cat = []
            
            fig = figure(x_range=cat, title='Topological Vector [dim = 0]', height=250, tools = tools)

            if len(topologicalVector_0) != 0:
                fig.vbar(x=cat, top=topologicalVector_0, width=0.8, color="darkblue", alpha=0.5)
            
            fig.xaxis.axis_label = "Element"
            fig.yaxis.axis_label = "Threshold"
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                topologicalVector_1 = vec.GetTopologicalVectorFeature(pd1, thres=st.session_state.TopologicalVectorThreshold)
                cat = [f'{i}' for i in range(len(topologicalVector_1))]
            else:
                topologicalVector_1 = []
                cat = []
            
            fig = figure(x_range=cat, title='Topological Vector [dim = 1]', height=250, tools = tools)
            
            if len(topologicalVector_1) != 0:
                fig.vbar(x=cat, top=topologicalVector_1, width=0.8, color="darkred", alpha=0.5)
            
            fig.xaxis.axis_label = "Element"
            fig.yaxis.axis_label = "Threshold"
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Topological Vector (PH0)', topologicalVector_0)
            CreateDownloadButton('Topological Vector (PH1)', topologicalVector_1)
            st.markdown('#')

        # ************************************************
        #         Computing Tropical Coordinates
        # ************************************************
        isPersTropCoordsChecked = False if visualizeAll else st.checkbox('Tropical Coordinates')

        if isPersTropCoordsChecked or visualizeAll:
            st.subheader("Tropical Coordinates")

            if len(pd0) != 0:
                persTropCoords_0 = vec.GetPersTropicalCoordinatesFeature(pd0)
                xrange = range(len(persTropCoords_0))
            else:
                persTropCoords_0 = []
                xrange = []
            
            fig = figure(title='Tropical Coordinates [dim = 0]', height=250, tools = tools)

            if len(persTropCoords_0) != 0:
                fig.vbar(x=xrange, top=persTropCoords_0, width=0.9, color="darkblue", alpha=0.5)
                fig.xaxis.major_label_overrides = {i: f'{i+1}' for i in range(len(persTropCoords_0))}
            
            fig.xaxis.axis_label = "Coordinate"
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                persTropCoords_1 = vec.GetPersTropicalCoordinatesFeature(pd1)
                xrange = range(len(persTropCoords_1))
            else:
                persTropCoords_1 = []
                xrange = []
            
            fig = figure(title='Tropical Coordinates [dim = 1]', height=250, tools = tools)
            
            if len(persTropCoords_1) != 0:
                fig.vbar(x=xrange, top=persTropCoords_1, width=0.9, color="darkred", alpha=0.5)
                fig.xaxis.major_label_overrides = {i: f'{i+1}' for i in range(len(persTropCoords_1))}

            fig.xaxis.axis_label = "Coordinate"
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Tropical Coordinates (PH0)', persTropCoords_0)
            CreateDownloadButton('Tropical Coordinates (PH1)', persTropCoords_1)
            st.markdown('#')

        # ************************************************
        #         Computing Template Function
        # ************************************************
        isTemplateFunctionChecked = False if visualizeAll else st.checkbox('Template Function')

        if isTemplateFunctionChecked or visualizeAll:
            st.subheader("Template Function")
            
            if len(pd0) != 0:
                templateFunc_0 = vec.GetTemplateFunctionFeature(pd0)[0]
                xrange = range(len(templateFunc_0))
            else:
                templateFunc_0 = []
                xrange = []
            
            fig = figure(title='Template Function [dim = 0]', height=250, tools = tools)

            if len(templateFunc_0) != 0:
                fig.vbar(x=xrange, top=templateFunc_0, width=0.9, color="darkblue", alpha=0.5)
                fig.xaxis.major_label_overrides = {i: f'{i+1}' for i in range(len(templateFunc_0))}
            
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                templateFunc_1 = vec.GetTemplateFunctionFeature(pd1)[0]
                xrange = range(len(templateFunc_1))
            else:
                templateFunc_1 = []
                xrange = []
            
            fig = figure(title='Template Function [dim = 1]', height=250, tools = tools)

            if len(templateFunc_1) != 0:
                fig.vbar(x=xrange, top=templateFunc_1, width=0.9, color="darkred", alpha=0.5)
                fig.xaxis.major_label_overrides = {i: f'{i+1}' for i in range(len(templateFunc_1))}
            
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Template Function (PH0)', templateFunc_0)
            CreateDownloadButton('Template Function (PH1)', templateFunc_1)

            st.markdown('#')

        # ************************************************
        #         Computing Adaptive Template System
        # ************************************************
        isTemplateSystemChecked = False if visualizeAll else st.checkbox('Adaptive Template System')

        if isTemplateSystemChecked or visualizeAll:
            st.subheader("Adaptive Template System")
            st.caption("Clustering method: GMM")

            if len(train_pd0s) > 0 and len(train_pd1s) > 0:
                st.slider("Number of Clusters", 1, 15, value=10, step=1, key='TempSysNComp')

                if len(pd0) != 0:
                    labels = [i for i in range(len(train_pd0s))]
                    TempSys_0 = vec.GetAdaptativeSystemFeature(train_pd0s, [pd0], labels, d=st.session_state.TempSysNComp)
                    cat = [f'{i}' for i in range(len(TempSys_0[0]))]
                else:
                    labels = []
                    TempSys_0 = []
                    cat = []
                
                fig = figure(x_range=cat, title='Adaptive Template System [dim = 0]', height=250, tools = tools)

                if len(TempSys_0) != 0:
                    fig.vbar(x=cat, top=TempSys_0[0], width=0.9, color="darkblue", alpha=0.5)
                
                fig.xaxis.axis_label = "Clusters"
                st.bokeh_chart(fig, use_container_width=True)

                if len(pd1) != 0:
                    labels = [i for i in range(len(train_pd1s))]
                    TempSys_1 = vec.GetAdaptativeSystemFeature(train_pd1s, [pd1], labels, d=st.session_state.TempSysNComp)
                    cat = [f'{i}' for i in range(len(TempSys_1[0]))]
                else:
                    labels = []
                    TempSys_1 = []
                    cat = []
                
                fig = figure(x_range=cat, title='Adaptive Template System [dim = 1]', height=250, tools = tools)

                if len(TempSys_1) != 0:
                    fig.vbar(x=cat, top=TempSys_1[0], width=0.9, color="darkred", alpha=0.5)
                
                fig.xaxis.axis_label = "Clusters"
                st.bokeh_chart(fig, use_container_width=True)
                
                TempSys_0_csv = TempSys_0[0] if len(TempSys_0) != 0 else []
                TempSys_1_csv = TempSys_1[0] if len(TempSys_1) != 0 else []

                CreateDownloadButton('Adaptive Template System (PH0)', TempSys_0_csv)
                CreateDownloadButton('Adaptive Template System (PH1)', TempSys_1_csv)
            else:
                st.error('''In order to compute Adaptive Template System features, you need to select at least 
                            one more data of the same type as the selected one. Use the second uploader from the sidebar on the left. 
                            If you have already done this step, so selected data did not produce any barcode, select another file.''')

    # Display error message if no file is selected 
    else:
        st.error("Please upload a file to start.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error("Exception occurred:\n", exc_info=True)

        # # creating/opening a file
        # f = open("log.txt", "a")
    
        # # writing in the file
        # f.write(f'{datetime.datetime.now()}\nException occurred:\n')
        # f.write(str(e) + '\n')
        # f.write(traceback.format_exc())
        # f.write('\n\n')
        
        # # closing the file
        # f.close()

        st.error(f'''An unexpected error occured. Please refresh you browser to restart. This could be due to wrong file format
                    or overflow due to large file size. Please use a smaller file in correct format and try again.\n\n{e}''')
