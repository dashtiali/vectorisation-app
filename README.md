istent Homology
Computation | Visualization | Vectorization

> This code accompanies the paper "A Survey of Vectorization Methods in Topological Data Analysis, Dashti A Ali, Aras Asaad, Maria Jose Jimenez, Vidit Nanda, Eduardo Paluzo-Hidalgo, and Manuel Soriano-Trigueros"


## About:

This web application can be used to compute and visualize persistence barcodes as well as the thirteen vectorization techniques
investigated in this paper. Beside running online, the companion web application can also
be downloaded and run on users local machines to compute and visualize barcodes from
large point-clouds and images. This no code application can be used as a convenient tool
to explore different vectorization techniques by beginners and experts alike. The WebApp
can be accessed in https://persistent-homology.streamlit.app.


The app is implemented in Python using Streamlit library together with other Python libraries. Streamlit is a Python framework for creating web applications for machine learning and data science projects. The application has a sidebar to choose between different types of input data and displays several options for data visualization and exportation. A sample image/point-cloud from each of the datasets used in this paper is preloaded into the web application together with their associated vectorisation techniques and visualizations. The user can either select from one of these  data samples or upload an image or point cloud file for which the different vectorization methods will be computed and visualized. Uploaded images by users will be resized to 64-rows by 64-columns and converted to gray-scale. The user can only upload 3D point cloud and only processing the first 100 points for their uploaded 3D point cloud. These restrictions are removed when the user run the application on their local machines. Furthermore, cubical complexes and Vietoris-Rips filtrations are used by default to build persistence barcodes for images and point clouds uploaded by the user,respectively.


The app will compute the persistence barcodes of dimension 0 and 1 and the user can select to have them plotted by ticking a check box (persistence diagram representation can also be displayed). All  barcode vectorization methods considered in this paper can be computed and visualized in different formats (tables, graphics, scattered plots), depending on the type of vectorizarion. Finally, an export button is associated with each barcode/diagram and vectorization method so that users can easily download associated data for further data analysis tasks.


## Run Locally
To run the app locally on your machine, first download this repository. Then follow these steps to run the app
on your local machine:

1. Extract the compressed files into a folder.
2. Open command window in windows or terminal in Mac/Linux.
3. Navigate to the folder where extracted files are.
4. Install ```streamlit``` package.
5. If you have all dependencies already installed skip step 6.
6. All dependencies for the app to run are located in requirements.txt file, 
to install all these dependencies in one go execute this command:
```
pip install -r requirements.txt
```
7. Then execute the following command to run the app:
```
streamlit run app.py local
```