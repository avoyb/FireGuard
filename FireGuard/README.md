# Project FireGuard: Wildfire Detection System

FireGuard is a project that leverages U-net deep learning model to identify wildfire burn scars from satellite images. The model was trained via Databricks and was implemented on Streamlit. A video walkthrough ↗ of the project is available.

# Project Structure
The following areas are elaborated in this document:

Setup Instructions
Rationale for FireGuard
Functional Capabilities
Creation Process
Illustrative Predictions
Contact Information
Setup Instructions

To get started with FireGuard, clone the repository and navigate to the ./app directory. Using a Python IDE, navigate to the /app directory and execute the command streamlit run app.py in your command line. Ensure you have installed the dependencies listed in the "requirement.txt" file.

The Streamlit application allows users to upload an image in "*.PNG" format for wildfire detection. The application will automatically render the raw image, estimate burn scar probability, and generate a predicted burn scar mask. Users can also select forest type and image resolution to estimate the burnt area and associated CO2 emission.

Rationale for FireGuard
Wildfires pose an immense threat to human life, property, and the environment. With a limited number of earth observation satellites capable of monitoring wildfires, there's a pressing need for more accessible tools. FireGuard addresses this need by leveraging deep learning technologies to enhance wildfire monitoring capabilities using satellite imagery.

Functional Capabilities
FireGuard is a user-friendly application that accepts satellite imagery and outputs forest fire probability and segmented burn scar zones. It also provides an estimate of the total area of the wildfire and the total CO2 emission, given the image resolution and forest type.

Creation Process
The development of FireGuard involved the following steps:

Satellite imagery was sourced using the Google image API to form the training dataset.
The burn scar zones were manually outlined to form labels.
A training pipeline was constructed, integrating a U-net model.
The model was trained on Databricks and the trained model was stored on AWS S3.
The Streamlit application was developed and deployed.
The CO2 emission estimation was based on existing research, namely the California Greenhouse Gas Emission for 2000 to 2017 Trends of Emissions and Other Indicators (2019) and The Fire INventory from NCAR (FINN): A high resolution global model to estimate the emissions from open burning (2011).

Illustrative Predictions
FireGuard was tested on various images, with the output including the raw image, manually added burn scar masks, burn scar probabilities projected by the model, and the final burn scar predictions from the model.

