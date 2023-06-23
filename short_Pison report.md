# Pison Report: 
Highlights: https://github.com/gerardoRO/Pison
* 6 gestures identified with K Means clustering
* XGBoost Classifier returns an F1 score of 84% in the classification.

### 1. Data Exploration
<b>Goal: Investigate underlying patterns and potential issues with the data. </b>
* Dataset has some big jumps in time that separate the body positionings, so first I looked at the IMU data in these groups to see if there were clear groupings of gestures.
* The IMU data also seems to be all over the place in terms of scaling, this could be due to new gestures present or due to noisiness in the data.
* The accelerometer data also seems to have some strong underlying oscillations, that perhaps can highlight repetitions of gestures, so we can use this to segregate gestures.

### 2. Dataset Generation
<b>Goal: Generate a NxM table with N number of samples ideally corresponding to a gesture.</b>
I decided to use the accelerometer underlying oscillations and peaks to segregate our datasets into samples. I worked with a fixed value of height and distance chosen, but I later on explore how changing these parameters affects our gesture classification.

### 3. Data Augmentation
<b> Goal: Generate the M properties of the table described above to classify gestures effectively. </b>

* I applied Singular Spectrum Analysis to the oscillatory signals (accelereometer, sensor data) and fourier analysis to the same signals
* Also decided to estimate the power of the SSA components and the Fourier spectra, as well as the time component of IMU data. For the time components I also estimated the mean and standard deviation.

### 4. Clustering
<b> Goal: Generate labels for the N samples that are reliable groupings of gestures. </b>

Visualizing the K Means results, we can see an increase in silhouette score at around 6-8 gestures, and an elbow like behavior around 6-7 for the inertia score.
![simple k means](images/kmeans_clustering.png)
![clustering](images/kmeans_clustering_visualization.png)
We see that when I segregated with a peak of 10 and a distance of 200, and 6 gestures, we get some consistent gesture motions. With linear motion, rotational motion,and some more diverse motion.
### 5. Classification
![ROC Curve](images/auc_curve_onevsall_classification.png)
For more in depth results, look [here](Pison_hw.ipynb), but we get a roc_score of 99%, and a f1 score of 84%. Not only that, but looking at the gestures predictions we have a varied predictions and the labels are not correlated with body movement label.