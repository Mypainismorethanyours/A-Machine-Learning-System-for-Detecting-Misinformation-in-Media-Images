
## A Machine Learning System for Detecting Misinformation in Media Images
# Model Training
[Link](https://github.com/Mypainismorethanyours/A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/tree/main/Finetune)

We fine-tune the Qwen2.5-VL-3B-Instruct pre-trained model. The input consists of a fixed question, "Is this image manipulated or synthesized?" along with the user-uploaded image, and the output is a text segment that includes the judgment result and possible reasons.

We support DeepSpeed for distributed training, MLFlow to track experiments, and Ray cluster to submit training jobs.

[Instruction to run](https://github.com/Mypainismorethanyours/A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/blob/main/Finetune/docker/README.md)

[One of Train Code](https://github.com/Mypainismorethanyours/A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/blob/main/Finetune/train_single_GPU_LoRA_Sample.py)


# Instructions on how to bring up the Flask app interface + FASTAPI docker containers 

on the node instance run : this uses a docker compose file to bring up the two containers for the

docker compose -f ~/A-Machine-Learning-System-for-Detecting-Misinformation-in-Media-Images/docker/docker-compose-fastapi.yaml up -d


sanity check to see all containers are running properly: 

docker compose ps

check logs for Jupyter container:
- This will give the token url  (http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXX)
- To  access the Jupiter container notebook environment in the browser replace 127.0.0.1 with the floating IP address of the reserved instance


The system optimizations that we tested on and then applied based on the results to a  trained model targeted reduction of the inference time of our model. However, the overall includes other delays notably system related delays.

Our system level ptimizations were implemneted such that theyb are not harcoded and flexible in that whenever we have a newly trained model we can test different optimizations for that new model, then based on the results we apply the optimizations that result in the best performance( lowest latency if thats a requiremnt that we set for our model). After we have a newly trained the optimization strategies pipeline is ran and the recommended optimizations based on the results of the tests are stored in a json file. When we do model inferencing the current model is loaded as well as the recommended optimization strategies are loaded from the json file and applied to the model for inferencing. 





#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->
Strategy:

Collect images from social media using APIs like Twitter API, Instagram Graph API, or through web scraping.
Use an ETL (Extract, Transform, Load) pipeline to preprocess the images for model training.
Store preprocessed images and model weights in persistent cloud storage.
Continuously gather new images for model re-training to ensure the system stays up-to-date with new AI generation techniques.
Relevant Diagram Parts:

Data collection API endpoints.
Data storage and management systems.
ETL pipeline for preprocessing images.
Justification:

Continuous collection and preprocessing of data ensures that the model is regularly updated with fresh, relevant examples of social media content.
Related to Course Material:

Data pipeline management meets Unit 8 (Data Pipeline) requirements, particularly around persistent storage and managing real-time data for inference.

#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->
Strategy:

Implement a CI/CD pipeline that automatically retrains the model with new data, evaluates it, and deploys the updated version to production.
Monitor performance in real-time and use this feedback to initiate model re-training if performance drops.
Relevant Diagram Parts:

CI/CD pipeline for training, evaluation, and deployment.
Model registry and version control for tracking model iterations.
Justification:

Automating the retraining process ensures that the model adapts to new types of AI-generated images over time, maintaining high accuracy in real-world applications.
Related to Course Material:

Automated pipelines align with Unit 3 (DevOps) for CI/CD, ensuring that the model is continuously updated with minimal manual intervention.


