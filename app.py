'''
    Name: Persistent Homology Vectorization Methods Web App
    This app is a part of the following paper:

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

# ************************************************
#   Necessary packages to run the application
# ************************************************
import streamlit as st
from PIL import Image
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import vectorisation as vec
import plotly.express as px
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.models import ColumnDataSource, Range1d
import pandas as pd
import io
import os
import sys
from scipy.spatial import distance
from ripser import ripser
import persim
import logging
# import traceback
# import datetime


# ************************************************
#           @st.cache, what does it do?
# According to Streamlit documentation:
# When you mark a function with Streamlit's cache annotation, it tells Streamlit 
# that whenever the function is called it should check three things:
#     1.The name of the function
#     2.The actual code that makes up the body of the function
#     3.The input parameters that you called the function with
# If this is the first time Streamlit has seen those three items, with those exact 
# values, and in that exact combination, it runs the function and stores the result
# in a local cache. Then, next time the function is called, if those three values have
# not changed Streamlit knows it can skip executing the function altogether. Instead,
# it just reads the output from the local cache and passes it on to the caller.
# ************************************************

@st.cache(ttl=1800, max_entries=20)
def load_image(file_path, resize=False, resize_val=64, convert_to_gray=True):
    '''
    Load image file and convert to grayscale
    :param file_path: full path of the image file
    :type file_path: string
    :return: numpy 2D array -- Grayscale version of the loaded image
    '''
    img = Image.open(file_path)

    if resize:
        img = img.resize((resize_val, resize_val))
    
    if convert_to_gray:
        img = img.convert("L")

    return img

@st.cache(ttl=1800, max_entries=10)
def load_point_cloud(file_path, limit_num_rows=True):
    '''
    Load point cloud from csv file
    :param file_path: full path of the csv file
    :type file_path: string
    :csv format: 3 columns without header
    :return: numpy 2D array -- A Nx3 pandas dataframe with ['x', 'y', 'z'] headers
    '''
    if limit_num_rows:
        df = pd.read_csv(file_path, names=['x', 'y', 'z'], nrows=100)
        df = df.drop_duplicates()
    else:
        df = pd.read_csv(file_path, names=['x', 'y', 'z'])

    return df

@st.cache(ttl=1800, max_entries=20)
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


@st.cache(ttl=1800, max_entries=20)
def GetPds(data, isPointCloud, dim=64):
    '''
	Compute persistence barcodes (H0, H1) from image or point cloud
	:param data: image or point cloud data
	:type data: numpy 2D array
    :param isPointCloud: indicates if the input data is a point cloud, otherwise it is an image
	:type isPointCloud: Boolean
	:return pd0, pd1:  numpy 2D arrays -- Persistence barcodes in dimension 0, 1
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
        data_gudhi = np.resize(data, [dim, dim])
        data_gudhi = data_gudhi.reshape(dim*dim,1)
        cub_filtration = gd.CubicalComplex(dimensions = [dim,dim], top_dimensional_cells=data_gudhi)
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
            np.savetxt(buffer, [row], delimiter=',') 
        st.download_button(
            label=f"Download {label}",
            data = buffer, # Download buffer
            file_name = f'{label.replace(" ", "_")}.csv',
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


def main(run_locally):
    '''
	The main function to create the Streamlit app
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
    # Fix KMeans memory leakage issue
    os.environ["OMP_NUM_THREADS"] = '1'
    
    # Populating page info and markups 
    st.set_page_config(page_title="Persistent Homology")
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.title("Persistent Homology")
    st.subheader('Computation | Visualization | Vectorization')
    st.info('This WebApp can be used to compute, visualize and featurize Persistent Homology Barcodes')

    # Populating sidebar and its main menu
    sidebar_menu = ["Cifar10","Fashion MNIST","Outex68", "Shrec14", "Custom"]
    choice = st.sidebar.selectbox("Select a Dataset", sidebar_menu)
    filtrationType = "Intensity values on cubical complex"

    # Set main and train files path
    mainPath = os.getcwd()
    train_folder = mainPath + r"/data/training_pds/"

    # Initialize or reset train file paths
    train_pd0_file_paths = []
    train_pd1_file_paths = []
    train_file_paths = []
    
    isPointCloud = False
    resize_image = True
    image_resize_val = 64

    # Populate sample file and the train files paths
    # for selected option from the sidebar main menu
    if choice == "Cifar10":
        file_path = mainPath + r"/data/cifar10.png"
        pd0_file_path = mainPath + r"/data/cifar10_ph0.csv"
        pd1_file_path = mainPath + r"/data/cifar10_ph1.csv"
        train_pd0_file_paths = [train_folder + "cifar30_ph0.csv", train_folder + "cifar51_ph0.csv"]
        train_pd1_file_paths = [train_folder + "cifar30_ph1.csv", train_folder + "cifar51_ph1.csv"]
    elif choice == "Fashion MNIST":
        file_path = mainPath + r"/data/FashionMNIST.jpg"
        pd0_file_path = mainPath + r"/data/FashionMNIST_ph0.csv"
        pd1_file_path = mainPath + r"/data/FashionMNIST_ph1.csv"
        train_pd0_file_paths = [train_folder + "FashionMNIST21_ph0.csv", train_folder + "FashionMNIST502_ph0.csv"]
        train_pd1_file_paths = [train_folder + "FashionMNIST21_ph1.csv", train_folder + "FashionMNIST502_ph1.csv"]
        filtrationType = "Edge growing"
    elif choice == "Outex68":
        file_path = mainPath + r"/data/Outex1.bmp"
        pd0_file_path = mainPath + r"/data/outex1_ph0.csv"
        pd1_file_path = mainPath + r"/data/outex1_ph1.csv"
        train_pd0_file_paths = [train_folder + "Outex2_ph0.csv", train_folder + "Outex3_ph0.csv"]
        train_pd1_file_paths = [train_folder + "Outex2_ph1.csv", train_folder + "Outex3_ph1.csv"]
    elif choice == "Shrec14":
        file_path = mainPath + r"/data/shrec14.png"
        pd0_file_path = mainPath + r"/data/shrec14_data_0_ph0.csv"
        pd1_file_path = mainPath + r"/data/shrec14_data_0_ph1.csv"
        train_pd0_file_paths = [train_folder + "shrec14_data_1_ph0.csv", train_folder + "shrec14_data_2_ph0.csv"]
        train_pd1_file_paths = [train_folder + "shrec14_data_1_ph1.csv", train_folder + "shrec14_data_2_ph1.csv"]
        filtrationType = "Heat Kernel Signature"
    else:
        #This is when the user is opt to select/use their own data to compute/visualize/features PH Barcodes.
        file_path = st.sidebar.file_uploader(f"Upload Image Or 3D Point Cloud.{'' if run_locally else ' For 3D point cloud data, only the first 100 points will be processed'}",
                                             type=['png','jpeg','jpg','bmp','csv'])
        if file_path is not None:
            isPointCloud = os.path.splitext(file_path.name)[1] == ".csv"

            if run_locally and not isPointCloud:
                col1, col2 = st.sidebar.columns(2)
                resize_image = col1.checkbox('Resize Image', value=True)
                image_resize_val = col2.number_input('Resizing value:', disabled=not resize_image, value=64, min_value=8, max_value=8000, step=1)

            selectedType = os.path.splitext(file_path.name)[1]
            train_file_paths = st.sidebar.file_uploader('''In order to compute Atol or Adaptive Template System features, you need to select 
                                                            some data samples  of the same type as the selected one.''',
                                                            type=[selectedType], accept_multiple_files=True)
    
    # ************************************************
    #              About Section (sidebar)
    # ************************************************
    st.sidebar.markdown('#')
    st.sidebar.caption('''### About:''')
    
    st.sidebar.caption('''
        This WebApp can be used to compute, visualize  and featurize Persistent Homology Barcodes. It is part of the following research paper:\\
        [Paper Reference](https://google.com)''')
    st.sidebar.markdown('''<div style="text-align: justify; font-size: 14px; color: rgb(163, 168, 184);">
        This WebApp (currently) can compute PH barcodes in dimension 0 and 1. The user can select corresponding check boxes to plot Persistence Barcodes 
        along with an option to plot their corresponding persistence diagrams in one plot. Furthermore, 12 barcode featurization methods can be selected to 
        compute and visualize as tables (e.g. Persistence statistics) or plots. Last but not least, an export button is associated 
        with each PH barcodes, diagrams and 12 featurized PH barcodes so that users can download associated data to their 
        local machines for further exploration.
    </div>''', unsafe_allow_html=True)

    # When file path is selected or changed the following will be populated
    if file_path is not None:
        
        # ************************************************
        #        Load and visualize the input file
        # ************************************************
        isShowImageChecked = st.checkbox('Input File', value=True)

        # Do not limit number of rows for point cloud when app is running locally
        limit_num_rows = False if run_locally else True

        if isPointCloud:
            filtrationType = "Vietoris-Rips"
        
        if isPointCloud:
            input_data = load_point_cloud(file_path, limit_num_rows)
        else:
            if choice == "Custom":
                input_data = load_image(file_path, resize_image, image_resize_val)
            else:
                as_gray = choice != "Shrec14"
                input_data = load_image(file_path, convert_to_gray=as_gray)

        if isShowImageChecked:
            if isPointCloud:
                #ploting the point cloud selected by the user
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
                if choice == "Shrec14":
                    st.image(input_data, width=400)
                else:
                    #show the image (from our sample images or custom selected image by the user)
                    col1, col2 = st.columns(2)
                    col1.image(input_data,width=250)
                    if choice == "Custom":
                        col2.markdown(
                            f"""
                            The following preprocessing applied to the uploaded image:
                            {f'- Resizing to {image_resize_val} rows by {image_resize_val} columns' if resize_image else ''}
                            - Converting to grayscale
                            """
                            )
            
        st.caption(f"Filtration type: {filtrationType}")
        
        # ************************************************
        # 1. Compute persistence barcodes of the input file 
        # 2. Load pre-computed persistence barcodes of 
        # the train data or Load selected train files
        # and compute their persistence barcodes
        # ************************************************

        #we start by initializing variables
        train_pd0s = list()
        train_pd1s = list()
        train_pd0s.clear()
        train_pd1s.clear()

        if choice == "Custom" and not isPointCloud:
            data = np.array(input_data)
            dimension = image_resize_val if resize_image else min(data.shape)
        else:
            data = input_data
            dimension = image_resize_val

        #when user selected sample data sets, load pre-computed PersBarcodes in dim-0 & dim-1
        if(choice == "Custom"):
            pd0, pd1 = GetPds(data, isPointCloud, dim=dimension)
        else:
            pd0 = load_csv(pd0_file_path)
            pd1 = load_csv(pd1_file_path)

        if (choice == "Custom"):
            # This is where the user can select more than one data sample for ATOL & ATS vectorization methods.
            # Then we compute persistence diagrams for selected data samples. 
            if train_file_paths is not None:
                for file in train_file_paths:
                    if isPointCloud:
                        file_data = load_point_cloud(file, limit_num_rows)
                    else:
                        file_data = load_image(file, resize_image, image_resize_val)
                        file_data = np.array(file_data)

                    file_pd0, file_pd1 = GetPds(file_data, isPointCloud, dim=dimension)

                    if len(file_pd0) != 0:
                        train_pd0s.append(file_pd0)
                    if len(file_pd1) != 0:
                        train_pd1s.append(file_pd1)
        else:
            # The following step will be executed to load pre-computed diagrams when the user opt to explore one of the data samples from the
            # three datasets of Outex, FMNIST and Shrec14 used in our paper. 
            train_pd0s = [load_csv(p) for p in train_pd0_file_paths]
            train_pd1s = [load_csv(p) for p in train_pd1_file_paths]
        
        # Populate radio buttons for visualize all or custom selected option
        visualizeAll = False
        visualizationMode = st.radio("Visualization Mode: ", ('Persistence Barcodes', 'Persistence Diagram', 'Persistence Statistics', 'Entropy Summary Function',
                                                              'Algebraic Functions', 'Tropical Coordinates', 'Complex Polynomial', 'Betti Curve', 
                                                              'Persistence Landscapes', 'Persistence Silhouette', 'Lifespan Curve', 'Persistence Image', 
                                                              'Template Function', 'Adaptive Template System', 'ATOL'), horizontal=True)
        st.markdown("""---""")

        # Initialize chart tools for bokeh chart type
        tools = ["reset", "pan","box_zoom", "wheel_zoom", "hover"]

        # Create a log function to transform array values for visualization
        log_func = lambda x : -np.log(1 + np.absolute(x)) if x < 0 else np.log(1 + x)
        # Apply log_func function to array. 
        log_func_arr = np.vectorize(log_func)
        
        # ************************************************
        #         Visualizing Persistence Barcodes
        # ************************************************
        if visualizationMode == 'Persistence Barcodes':
            st.subheader("Persistence Barcodes")

            # Visualizing PersBarcode in dim-0 (Together with its error handling)
            if len(pd0) != 0:
                source = ColumnDataSource(data={'birth': pd0[:,0], 'death': pd0[:,1], 'y': range(len(pd0[:,0]))})
            else:
                # This is an error handling step so that when the PersBarcode data is empty, 
                # then initialize an empty array and return an empty plot to avoid getting an error
                # A Similar approach has been repeated for the rest of the vectorization method visualization.
                source = ColumnDataSource(data={'birth': [], 'death': [], 'y': []})
            
            fig = figure(title='Persistence Barcode [dim = 0]', height=250, tools = tools)
            fig.hbar(y='y', left ='birth', right='death', height=0.1, alpha=0.7, source=source)
            fig.yaxis.visible = False

            if len(pd0) == 1:
                fig.y_range = Range1d(-1, 1)
            
            st.bokeh_chart(fig, use_container_width=True)

            # Visualizing PersBarcode in dim-1
            if len(pd1) != 0:
                source = ColumnDataSource(data={'birth': pd1[:,0], 'death': pd1[:,1], 'y': range(len(pd1[:,0]))})
            else:
                source = ColumnDataSource(data={'birth': [], 'death': [], 'y': []})
            fig = figure(title='Persistence Barcode [dim = 1]', height=250, tools = tools)
            fig.hbar(y='y', left ='birth', right='death', height=0.1, color="darkorange", alpha=0.7, source=source)
            fig.yaxis.visible = False

            if len(pd1) == 1:
                fig.y_range = Range1d(-1, 1)
            
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('PH barcode dim0', pd0)
            CreateDownloadButton('PH barcode dim1', pd1)
            st.markdown('#')

        # ************************************************
        #         Visualizing Persistence Diagram
        # ************************************************
        if visualizationMode == 'Persistence Diagram':
            st.subheader("Persistence Diagram")
            # dgms = []

            # if len(pd0) != 0:
            #     dgms.append(pd0)
            
            # if len(pd1) != 0:
            #     dgms.append(pd1)

            col1, col2 = st.columns(2)

            fig, ax = plt.subplots()

            if len(pd0) != 0:
                persim.plot_diagrams([pd0],labels= ["H0"], ax=ax)
            
            ax.set_title("Persistence Diagram H0")
            col1.pyplot(fig)

            fig, ax = plt.subplots()
            
            if len(pd1) != 0:
                persim.plot_diagrams([pd1],labels= ["H1"], ax=ax)
            
            ax.set_title("Persistence Diagram H1")
            col2.pyplot(fig)
        # Create a download button for PersDiagram plot, PLease.

        # ************************************************
        # Computing and Visualizing Persistence Statistics
        # ************************************************
        if visualizationMode == 'Persistence Statistics':
            st.subheader("Persistence Statistics")
            
            stat_0 = vec.GetPersStats(pd0)
            stat_1 = vec.GetPersStats(pd1)
            df = pd.DataFrame(np.array((stat_0[0:6], stat_1[0:6])), index=['PH(0)', 'PH(1)'])
            df.columns =['Birth Average', 'Death Average', 'Birth STD.', 'Death STD.', 
                         'Birth Median', 'Death Median']
            st.dataframe(df, use_container_width=True)

            df = pd.DataFrame(np.array((stat_0[6:11], stat_1[6:11])), index=['PH(0)', 'PH(1)'])
            df.columns =['Bar Length Average', 'Bar Length STD', 
                         'Bar Length Median', 'Bar Count', 'Persistence Entropy']
            st.dataframe(df, use_container_width=True)

            CreateDownloadButton('Persistence Stats dim0', stat_0)
            CreateDownloadButton('Persistence Stats dim1', stat_1)
            st.markdown('#')
        
        # ************************************************
        # Computing and Visualizing Entropy Summary Function
        # ************************************************
        if visualizationMode == 'Entropy Summary Function':
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
            ApplyHatchPatternToChart(fig)
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                PersEntropy_1 = vec.GetPersEntropyFeature(pd1, res=st.session_state.PersEntropyRes)
                source = ColumnDataSource(data={'x': range(0, len(PersEntropy_1)), 'y': PersEntropy_1})
            else:
                PersEntropy_1 = []
                source = ColumnDataSource(data={'x': [], 'y': []})
            
            fig = figure(x_range=(0, len(PersEntropy_1)),title='Entropy Summary Function [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='red', alpha=0.5, line_width=2, source=source)
            ApplyHatchPatternToChart(fig)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Entropy Summary Function dim0', PersEntropy_0)
            CreateDownloadButton('Entropy Summary Function dim1', PersEntropy_1)
            st.markdown('#')

        # ************************************************
        #   Computing and Visualizing Algebraic Functions
        # ************************************************
        if visualizationMode == 'Algebraic Functions':
            st.subheader("Algebraic Functions")
            st.caption("Number of features = 5, Log values are displayed for visualization")
            
            fig = figure(title='Algebraic Functions [dim = 0]', height=250, tools = tools)

            if len(pd0) != 0:
                carlsCoords_0 = vec.GetCarlssonCoordinatesFeature(pd0)
                fig.vbar(x=range(len(carlsCoords_0)), top=log_func_arr(carlsCoords_0), width=0.9, color="darkblue", alpha=0.5)
            else:
                carlsCoords_0 = []
            
            fig.xaxis.major_label_overrides = {i: f'f{i+1}' for i in range(len(carlsCoords_0))}
            st.bokeh_chart(fig, use_container_width=True)

            fig = figure(title='Algebraic Functions [dim = 1]', height=250, tools = tools)

            if len(pd1) != 0:
                carlsCoords_1 = vec.GetCarlssonCoordinatesFeature(pd1)
                fig.vbar(x=range(len(carlsCoords_1)), top=log_func_arr(carlsCoords_1), width=0.9, color="darkred", alpha=0.5)
            else:
                carlsCoords_1 = []
            
            fig.xaxis.major_label_overrides = {i: f'f{i+1}' for i in range(len(carlsCoords_1))}
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Algebraic Functions dim0', carlsCoords_0)
            CreateDownloadButton('Algebraic Functions dim1', carlsCoords_1)
            st.markdown('#')

        # ************************************************
        #   Computing and Visualizing Tropical Coordinates
        # ************************************************
        if visualizationMode == 'Tropical Coordinates':
            st.subheader("Tropical Coordinates")
            st.caption("Log values are displayed for visualization")

            if len(pd0) != 0:
                persTropCoords_0 = vec.GetPersTropicalCoordinatesFeature(pd0)
                xrange = range(len(persTropCoords_0))
            else:
                persTropCoords_0 = []
                xrange = []
            
            fig = figure(title='Tropical Coordinates [dim = 0]', height=250, tools = tools)

            if len(persTropCoords_0) != 0:
                fig.vbar(x=xrange, top=log_func_arr(persTropCoords_0), width=0.9, color="darkblue", alpha=0.5)
                fig.xaxis.major_label_overrides = {i: f'F{i+1}' for i in range(len(persTropCoords_0))}
            
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
                fig.vbar(x=xrange, top=log_func_arr(persTropCoords_1), width=0.9, color="darkred", alpha=0.5)
                fig.xaxis.major_label_overrides = {i: f'F{i+1}' for i in range(len(persTropCoords_1))}

            fig.xaxis.axis_label = "Coordinate"
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Tropical Coordinates dim0', persTropCoords_0)
            CreateDownloadButton('Tropical Coordinates dim1', persTropCoords_1)
            st.markdown('#')

        # ************************************************
        #   Computing and Visualizing Complex Polynomial
        # ************************************************
        if visualizationMode == 'Complex Polynomial':
            st.subheader("Complex Polynomial")
            st.caption("Resolution = 10, Log values are displayed for visualization")
            st.selectbox("Polynomial Type",["R", "S", "T"], index=0, key='CPType')

            if len(pd0) != 0:
                CP_pd0 = vec.GetComplexPolynomialFeature(pd0, pol_type=st.session_state.CPType)
                coef = [f'{i}' for i in range(len(CP_pd0))]
            else:
                CP_pd0 = []
                coef = []
            
            fig = figure(x_range=coef, title='Complex Polynomial [dim = 0]', height=250, tools=tools)

            if len(CP_pd0) != 0:
                source = ColumnDataSource(data = {'coef'      : coef,
                                                  'Real'      : log_func_arr(CP_pd0[:,0]),
                                                  'Imaginary' : log_func_arr(CP_pd0[:,1])})
                
                fig.vbar(x=dodge('coef', -0.25, range=fig.x_range), top='Real', width=0.2, source=source,
                color="#c9d9d3", legend_label="Real")

                fig.vbar(x=dodge('coef',  0.0,  range=fig.x_range), top='Imaginary', width=0.2, source=source,
                color="#718dbf", legend_label="Imaginary")
            
            fig.x_range.range_padding = 0.1
            fig.xgrid.grid_line_color = None
            fig.legend.location = "top_left"
            fig.legend.orientation = "horizontal"
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                CP_pd1 = vec.GetComplexPolynomialFeature(pd1, pol_type=st.session_state.CPType)
                coef = [f'{i}' for i in range(len(CP_pd1))]
            else:
                CP_pd1 = []
                coef = []
            
            fig = figure(x_range=coef, title='Complex Polynomial [dim = 1]', height=250, tools = tools)

            if len(CP_pd1) != 0:
                source = ColumnDataSource(data = {'coef'      : coef,
                                                'Real'      : log_func_arr(CP_pd1[:,0]),
                                                'Imaginary' : log_func_arr(CP_pd1[:,1])})
                
                fig.vbar(x=dodge('coef', -0.25, range=fig.x_range), top='Real', width=0.2, source=source,
                color="#c9d9d3", legend_label="Real")

                fig.vbar(x=dodge('coef',  0.0,  range=fig.x_range), top='Imaginary', width=0.2, source=source,
                color="#718dbf", legend_label="Imaginary")
            
            fig.x_range.range_padding = 0.1
            fig.xgrid.grid_line_color = None
            fig.legend.location = "top_left"
            fig.legend.orientation = "horizontal"
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Complex Polynomial dim0', CP_pd0)
            CreateDownloadButton('Complex Polynomial dim1', CP_pd1)
            st.markdown('#')

        # ************************************************
        #       Computing and Visualizing Betti Curve
        # ************************************************
        if visualizationMode == 'Betti Curve':
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
            ApplyHatchPatternToChart(fig)
            st.bokeh_chart(fig, use_container_width=True)

            if len(pd1) != 0:
                Btt_1 = vec.GetBettiCurveFeature(pd1, st.session_state.BettiCurveRes)
                source = ColumnDataSource(data={'x': range(0, len(Btt_1)), 'y': Btt_1})
            else:
                Btt_1 = []
                source = ColumnDataSource(data={'x': [], 'y': Btt_1})
            
            fig = figure(x_range=(0, len(Btt_1)), title='Betti Curve [dim = 1]', height=250, tools = tools)
            fig.line(x='x', y='y', color='red', alpha=0.5, line_width=2, source=source)
            ApplyHatchPatternToChart(fig)
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Betti Curve dim0', Btt_0)
            CreateDownloadButton('Betti Curve dim1', Btt_1)
            st.markdown('#')

        # ************************************************
        # Computing and Visualizing Persistence Landscapes
        # ************************************************
        if visualizationMode == 'Persistence Landscapes':
            st.subheader("Persistence Landscapes")
            st.caption("Resolution = 100")
            st.slider("Number of landscapes", 1, 20, value=10, step=1, key='NumLand')
            fig, ax = plt.subplots()
            fig.set_figheight(3)
            resolution = 100
            numLand = st.session_state.NumLand

            if len(pd0) != 0:
                PL_0 = vec.GetPersLandscapeFeature(pd0, num=st.session_state.NumLand, res=resolution)

                for i in range(numLand):
                    ax.plot(PL_0[i*resolution:(i+1)*resolution])
            else:
                PL_0 = []
            
            ax.set_title("Persistence Landscapes [dim = 0]")
            fig.tight_layout()
            st.pyplot(fig)

            fig, ax = plt.subplots()
            fig.set_figheight(3)
            
            if len(pd1) != 0:
                PL_1 = vec.GetPersLandscapeFeature(pd1, num=st.session_state.NumLand, res=resolution)

                for i in range(numLand):
                    ax.plot(PL_1[i*resolution:(i+1)*resolution])
            else:
                PL_1 = []
            
            ax.set_title("Persistence Landscapes [dim = 1]")
            fig.tight_layout()
            st.pyplot(fig)

            CreateDownloadButton('Persistence Landscapes dim0', PL_0)
            CreateDownloadButton('Persistence Landscapes dim1', PL_1)

            st.markdown('#')

        # ************************************************
        # Computing and Visualizing Persistence Silhouette
        # ************************************************
        if visualizationMode == 'Persistence Silhouette':
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

            CreateDownloadButton('Persistence Silhouette dim0', PersSil_0)
            CreateDownloadButton('Persistence Silhouette dim1', PersSil_1)
            st.markdown('#')

        # ************************************************
        #   Computing and Visualizing Lifespan Curve
        # ************************************************
        if visualizationMode == 'Lifespan Curve':
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

            CreateDownloadButton('Lifespan Curve dim0', persLifeSpan_0)
            CreateDownloadButton('Lifespan Curve dim1', persLifeSpan_1)
            st.markdown('#')

        # ************************************************
        #   Computing and Visualizing Persistence Image
        # ************************************************
        if visualizationMode == 'Persistence Image':
            st.subheader("Persistence Image")
            st.caption("Bandwidth of the Gaussian kernel = 1, Weight function = Constant (i.e lambda x:1)")
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

            CreateDownloadButton('Persistence Image dim0', PI_0)
            CreateDownloadButton('Persistence Image dim1', PI_1)
            st.markdown('#')

        # ************************************************
        #   Computing and Visualizing Template Function
        # ************************************************
        if visualizationMode == 'Template Function':
            st.subheader("Template Function")
            st.caption("Number of a bins in each axis = 5, Padding = 2")
            
            if len(pd0) != 0:
                templateFunc_0 = vec.GetTemplateFunctionFeature(barcodes_train=[pd0], barcodes_test=[], d=5, padding=2)[0]

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
                templateFunc_1 = vec.GetTemplateFunctionFeature(barcodes_train=[pd1], barcodes_test=[], d=5, padding=2)[0]
                xrange = range(len(templateFunc_1))
            else:
                templateFunc_1 = []
                xrange = []
            
            fig = figure(title='Template Function [dim = 1]', height=250, tools = tools)

            if len(templateFunc_1) != 0:
                fig.vbar(x=xrange, top=templateFunc_1, width=0.9, color="darkred", alpha=0.5)
                fig.xaxis.major_label_overrides = {i: f'{i+1}' for i in range(len(templateFunc_1))}
            
            st.bokeh_chart(fig, use_container_width=True)

            CreateDownloadButton('Template Function dim0', templateFunc_0)
            CreateDownloadButton('Template Function dim1', templateFunc_1)

            st.markdown('#')

        # ************************************************
        # Computing and Visualizing Adaptive Template System
        # ************************************************
        if visualizationMode == 'Adaptive Template System':
            st.subheader("Adaptive Template System")
            st.caption("Clustering method: GMM")

            if len(train_pd0s) > 0 and len(train_pd1s) > 0:
                st.slider("Number of Clusters", 1, 15, value=5, step=1, key='TempSysNComp')

                if len(pd0) != 0:
                    TempSys_0 = vec.GetAdaptativeSystemFeature(train_pd0s, [pd0], d=st.session_state.TempSysNComp)
                    cat = [f'{i}' for i in range(len(TempSys_0[0]))]
                else:
                    TempSys_0 = []
                    cat = []
                
                fig = figure(x_range=cat, title='Adaptive Template System [dim = 0]', height=250, tools = tools)

                if len(TempSys_0) != 0:
                    fig.vbar(x=cat, top=TempSys_0[0], width=0.9, color="darkblue", alpha=0.5)
                
                fig.xaxis.axis_label = "Clusters"
                st.bokeh_chart(fig, use_container_width=True)

                if len(pd1) != 0:
                    TempSys_1 = vec.GetAdaptativeSystemFeature(train_pd1s, [pd1], d=st.session_state.TempSysNComp)
                    cat = [f'{i}' for i in range(len(TempSys_1[0]))]
                else:
                    TempSys_1 = []
                    cat = []
                
                fig = figure(x_range=cat, title='Adaptive Template System [dim = 1]', height=250, tools = tools)

                if len(TempSys_1) != 0:
                    fig.vbar(x=cat, top=TempSys_1[0], width=0.9, color="darkred", alpha=0.5)
                
                fig.xaxis.axis_label = "Clusters"
                st.bokeh_chart(fig, use_container_width=True)
                
                TempSys_0_csv = TempSys_0[0] if len(TempSys_0) != 0 else []
                TempSys_1_csv = TempSys_1[0] if len(TempSys_1) != 0 else []

                CreateDownloadButton('Adaptive Template System dim0', TempSys_0_csv)
                CreateDownloadButton('Adaptive Template System dim1', TempSys_1_csv)
            else:
                st.error('''In order to compute Adaptive Template System features, you need to select at least 
                            one more data of the same type as the selected one. Use the second uploader from the sidebar on the left. 
                            If you have already done this step, so selected data did not produce any barcode, select another file.''')

        # ************************************************
        #           Computing and Visualizing ATOL
        # ************************************************
        if visualizationMode == 'ATOL':
            st.subheader("ATOL")

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
                
                fig = figure(x_range=cat, title='ATOL [dim = 0]', height=250, tools = tools)
                
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
                
                fig = figure(x_range=cat, title='ATOL [dim = 1]', height=250, tools = tools)

                if len(atol_1) != 0:
                    fig.vbar(x=cat, top=atol_1[0], width=0.9, color="darkred", alpha=0.5)

                fig.xaxis.axis_label = "Clusters"
                st.bokeh_chart(fig, use_container_width=True)

                CreateDownloadButton('ATOL dim0', atol_0[0])
                CreateDownloadButton('ATOL dim1', atol_1[0])
            else:
                st.error('''In order to compute ATOL features, you need to select at least 
                            one more data of the same type as the selected one. Use the second uploader from the sidebar on the left. 
                            If you have already done this step, so selected data did not produce any barcode, select another file.''')
            
            st.markdown('#')

        # ************************************************
        #   Computing and Visualizing Topological Vector
        # ************************************************
        # isTopologicalVectorChecked = False if visualizeAll else st.checkbox('Topological Vector')

        # if isTopologicalVectorChecked or visualizeAll:
        #     st.subheader("Topological Vector")
        #     st.slider("Threshold", 2, 10, value=8, step=1, key='TopologicalVectorThreshold')

        #     if len(pd0) != 0:
        #         topologicalVector_0 = vec.GetTopologicalVectorFeature(pd0, thres=st.session_state.TopologicalVectorThreshold)
        #         cat = [f'{i}' for i in range(len(topologicalVector_0))]
        #     else:
        #         topologicalVector_0 = []
        #         cat = []
            
        #     fig = figure(x_range=cat, title='Topological Vector [dim = 0]', height=250, tools = tools)

        #     if len(topologicalVector_0) != 0:
        #         fig.vbar(x=cat, top=topologicalVector_0, width=0.8, color="darkblue", alpha=0.5)
            
        #     fig.xaxis.axis_label = "Element"
        #     fig.yaxis.axis_label = "Threshold"
        #     st.bokeh_chart(fig, use_container_width=True)

        #     if len(pd1) != 0:
        #         topologicalVector_1 = vec.GetTopologicalVectorFeature(pd1, thres=st.session_state.TopologicalVectorThreshold)
        #         cat = [f'{i}' for i in range(len(topologicalVector_1))]
        #     else:
        #         topologicalVector_1 = []
        #         cat = []
            
        #     fig = figure(x_range=cat, title='Topological Vector [dim = 1]', height=250, tools = tools)
            
        #     if len(topologicalVector_1) != 0:
        #         fig.vbar(x=cat, top=topologicalVector_1, width=0.8, color="darkred", alpha=0.5)
            
        #     fig.xaxis.axis_label = "Element"
        #     fig.yaxis.axis_label = "Threshold"
        #     st.bokeh_chart(fig, use_container_width=True)

        #     CreateDownloadButton('Topological Vector dim0', topologicalVector_0)
        #     CreateDownloadButton('Topological Vector dim1', topologicalVector_1)
        #     st.markdown('#')

    # Display error message if no file is selected 
    else:
        st.error("Please upload a file to start.")

if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1 and args[1] == "local":
        run_locally = True
    else:
        run_locally = False
    
    try:
        main(run_locally)
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

        st.error(f'''An unexpected error occurred. Please refresh you browser to restart. This could be due to wrong file format
                    or overflow due to large file size. Please use a smaller file in correct format and try again.\n\n{e}''')