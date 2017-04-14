
This project is to explore the existence of Underrated Answers in StackOverflow, the reason behind the underrating and to find whether it is possible to detect them automatically

<b>Underrated Answer</b> - The best solution at the time when I collect the data, but it has less votes than the top voted answer.

*******************************************************************************

OVERALL APPROACH

1. Data Collection
* Extratced 100,000+ StackOverflow (SO) urls from StackExahnge API
* Automatically found 2000+ SO posts with 3+ answers, and generated 2000+ JSON file to record Question, Answer, Comments data for each post
* Manually label the urls with "Y" or "N" to indicate whether there is an underrated answer. If it's "Y", reocrd the first 5 words of the Underrated Answer.

2. Feature Generation
* The features are generated for each answer of a question, 46 code & metrics, 12 sentiment metrics and 3 other metrics
* After generating the features, I generated 1 single CSV which contains features, the label IsUnderrated. Each row, the format is: QuestionID, AnswerID, ...61 features..., IsUnderrated

3. Data Analysis
* Data Preprocessing
* Clustering, to find group patterns
* Classification, to find whether it is possible to detect underrated answers automatically


*******************************************************************************

CODE DETAILS

* Data Collection
  * 100,000+ question list: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Predict_StackOverflow_Underrated_Answers/question_list.txt.zip
  * Find SO posts that contain 3+ answers, and generate a JSON file for each post: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Predict_StackOverflow_Underrated_Answers/SO_data_extraction.py
  * JSON format
    * Each JSON file is for 1 post, it contains question data (id, text, votes, favorite count, code)
    * It also contains all the answers for the question, each answer data has (id, text, votes, code)
    * It also contains all the comments for each answer, each comment data has (id, text, code, votes)
  * labeled urls: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Predict_StackOverflow_Underrated_Answers/labeled_urls_120.csv
  
* Feature Generation
  * 46 Code & Bug Metrics, 12 sentiment metrics, 3 other metrics: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Predict_StackOverflow_Underrated_Answers/generate_metrics.py
  * Code & Bug Metrics, used Radon and Pylint
  * Sentiment Metrics, Stanford CoreNLP sentiment analysis, analysis on each sentence and then count for each sentiment level (VeryPositive, Positive, etc.)
  * Answer code and Question code sequence match, the algorithm will ignore the junk items in sequences, and try to match the longest length
  * Measure MaxVotes - Vote, Vote/Total Votes in a post, because I was wondering if an answer's vote is much less than the highest votes in the post or if its votes occupies smaller percentage, would it be less likely to be underrated
  * The code is also responsible to aggregate JSON data into 1 CSV file, labeled data need to use the first 5 words to match the right answer. Finally each row in CSV is fmrmated in: QuestionID, AnswerID, ...61 features..., IsUnderrated
  
* Data Analysis & Insights
  * All code: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Predict_StackOverflow_Underrated_Answers/data_analysis_all_code.R
  * All the data is 442 rows (very small), 23 "Y" (imbalanced)
  * Data Preprocessing
    * remove zero variance data
    * check outliers, in this case, didn't remove/change outliers, because of the small data amount and the meaning of those feature values
    * remove highly correlated data
    * normalize features: scale to [0,1] and used KNN to normalize (it's another normalization method), later it turns out that scaling to [0,1] is better in this case
  * Clustering
    * K-Means, with Elbow methods to decode optimal cluster numbers
    * Clustering Ensembling, using hierarchical clusering methods ("ward.D", "single", "complete", "average", "mcquitty"), with hierarchical visualization, you can tell optimal number of clusters
    * After clustering, I chekced clusters distribution in both "Y", "N" classes, same patterns, so clustering in this case could not help find group patterns, since it cannot tell the difference between "Y" and "N" class
  * Classification
    * 67% training data, 23% testing data
    * Round 1 - Only use preprocessed data, Random Forests, SVM, GBM, XGBoost, all with the same cross validation settings, they predicted all as "N", which is not helpful
    * Round 2 - ROSE oversampling on training data, then Random Forests, SVM, GBM, XGBoost, all with the same cross validation settings, dind't work well either
    * Round 3 - ROSE on all the data, Random Forests, SVM, GBM, XGBoost, all with the same cross validation settings, SVM got higher than 80% Specificity, Sensitivity and Balanced Accuracy; Random Forests, XGBoost all got 1. The reason for this insane resullt is because the dataset is small
    * Round 4 - SMOTE on training data, then Random Forests, SVM, GBM, XGBoost, all with the same cross validation settings, SMOTE worked best with GBM, but showing imbalanced prediction results
    * Feature Selection - Boruta selected features + SVM, very high prediction accuracy
    * Feature Selection - Random Forests in Round 3 selected features + SVM, high prediction results but lower than Boruta
    * The reasons I use ROSE on all the data but SMOTE only on training data, is because ROSE makes duplications of subset of minority class while SMOTE generates similar data which may not exist in the original data, if use it on testing data, cannot guarantee the data is ground truth
    * Boruta is a wrapper of random forests, but it is an extension of random forests by comparing features' releavnce to prediction accuracy, theoretically, it should work better
    * Both Randomforests and Boruta has chosen 7 top features which are the same but with different rankings, 6 of them are related to coding style and they are all 0 for "Y" but have various values for "N". The 7th is about sentiment, which shows similar distribution in "Y" and "N". 
    * The insights I have gained is, based on this very small data set, it is possible to predict underrated answers, the reason behind the underrating is more about coding style than sentiment.


*******************************************************************************

FINAL PPT & PAPER

* Final Presentation (the presenattion is held to show progress, no need to finish the project at that time): https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Predict_StackOverflow_Underrated_Answers/Predict%20StackOverflow%20Underrated%20Answers.pdf
* <b>Final Paper</b>: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Predict_StackOverflow_Underrated_Answers/886_final_project_Hanhan_Wu.pdf
