# DAM: Diffusion Activation Maximization for 3D Global Explanations

A method for generating global explanations of point clouds based on DDPM models.
Usage:


1. Download dataset (ModelNet40, ShapeNet or other), put it in the /data folder.

2. Train the classifier to be explained by running train_classifier.py. For convenience, train_noised_classifier.py can be executed at the same time to train the noisified classifier, which improves performance during the explanation sampling process.

3. Train the PDT model by running train_PDT.py, which is the backbone of DDPM for generating highly perceptual explanations.

4. Run DAM_gen_one_sample.py or DAM_gen_batch.py to generate one or a batch of explanations, respectively.
