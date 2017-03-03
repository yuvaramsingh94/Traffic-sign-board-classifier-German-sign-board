# Traffic-sign-board-classifier-German-sign-board

the code in this repository used CNN to classify german trafic signs . it can classify 42 types of trafic signs . this model can also be used to search for trafic sign in a video stream .


## libraries used (python)
  * Tensor flow
  * numpy
  * open cv
  * matplotlib

## To run this Program
  install the needed libraries . an easy way to do is to download the miniconda for your os and follow these steps provided in this link
  * https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md
  
  ### steps
      * download CarND-Term1-Starter-Kit
      * choose between anaconda or docker 
        * #### for anaconda:
              https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md
        * #### for docker
              https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_docker.md
  * after installation , open terminal or cmd prompt and run jupyter with this command
      * jupyter notebook
  * this will open your default web browser . 
  * select the +++++++++++++++++++++++++++++++++++++++++.ipynb file . 
  * go to cell -> Run all

## Pipeline of program
  the working of the program can be broken down to following steps
  1. import the needed libraries
  2. import the signnames.csv file to form a dictionary which maps the sign names to the class number provided in y_labels 
  3. import the train , valid , test data from the pickle file
  4. summerize and visualize the input data 
  5. convert BGR image to Gray scale
  6. create additional training data randomly by applying image transformation technique to the training data
  7. define function conv2d() and pooling() to make the process of creating convolution layers much easier
  8. create weights and biases needed for the convolution layers and fully connected layers. use truncated_normal to generate random values
  9. for making the process of creating graph easier , define a function Net() to create the graph 
  10. create the placeholder names for the image data and labels  
  11. define prediction , accuracy , optimizer , loss 
  12. Run the model on the training data and save the session 
  13. use testing data to find the accuracy of our model
  14. to make things interesting , search for trafic singnals online and use thise to check our model
  15. convert the new images to fit out model by applying resize and grayscape methods 
  16. run the prediction on these images
  
  
  
  
  
  
  
  
## graph Layeres
      * convolution layer 1 with max pooling (32X32 -> 28X28 -> 24X24) depth(1 -> 6)
      * convolution layer 2 (24X24 -> 12X12) depth(6 -> 16)
      * convolution layer 3 with max pooling (12X12 -> 10X10 -> 5X5) depth (16 -> 32)
      * flatten the layer
      * fully connected layer 1
      * fully connected layer 2
      * dropout
      * output layer 
 
