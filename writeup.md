## Claire Kiekhaefer kiekhaef@usc.edu

## **Description**
This project classifies email body text as phishing or non-phishing by finetuning the transformer-based natural language processing model DistilBERT. 


**Final_Training_DistilBERT.ipynb** - contains the final organized code, training, and results.

**unfinished_test_Training_DistilBERT.ipynb** - contains an early training attempt to test hyperparameter values. It is not as important to view.


Computing power was very limited for this project. Due to this significant factor, Google Colab was utilized for its free GPU. This introduced challenges related to runtime session limits, which required saving trained models and results externally to prevent data loss. Additional design choices (like using DistilBERT instead of full BERT) were made to optimize training time while preserving strong model performance.


**[The Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)**

The email phishing dataset for this project consisted of six CSV files from various public sources: Enron, CEAS, Nazario, Nigerian Fraud, Ling, and SpamAssassin datasets. Additionally, a phishing_email.csv file combined body text and all additional context provided in four of the initial six datasets. The focus of this project was solely the body text of the email. All six datasets included the body text, but some also included additional aspects, such as "senders" or "subject lines," which were combined into the single text set phishing_email.csv. To overcome this, I took the body text from all six CSVs and combined them into a single set. I also retained the labeling column, which was consistent across all the datasets. This choice would enable the model to train on a wider range of sources while still focusing on the body text for model analysis. 

The final dataset included approximately 82,500 emails, 42,891 were spam emails, and 39,595 were legitimate emails. The data set was very balanced and did not require additional weighting.

Each email had a provided label:
- **1**: Phishing  (scam or data theft)
- **0**: Non-phishing (legitimate)

Only the email body text and label fields were used for training and evaluation.

## **Preprocessing the data**

There were several steps for preprocessing and cleaning up the email body text:

- Converted all text to lowercase for consistency.

- Replaced URLs and email addresses with placeholder tokens (`<URL>`, `<EMAIL>`) to reduce noise and character count.

- Removed empty or invalid emails with extra white space.

- Added a maximum character length of 20,000. 
       Although DistilBERT truncated inputs with its maximum token length of 512, additional truncation of the body text significantly improved preprocessing time for tokenizing each piece of body text. Some of the email bodies were extraordinarily long and 20,000 characters was a generous threshold based on the dataset statistics calculated below:

              === CHARACTER LENGTH STATS ===
              Number of emails : 82486
              Mean chars:   1760.4
              Median chars: 735.0
              90th pct:     3895.0
              95th pct:     5133.8
              99th pct:     12819.7

              === CHARACTER THRESHOLDS ===
              12,000 chars: 918 (1.11%)
              20,000 chars: 419 (0.51%)


- Tokenized text using the DistilBERT tokenizer with a maximum sequence length of 256 tokens. 
       512 is the maximum sequence length for DistilBERT. In the initial test run (unfinished_test_Training_DistilBERT), a maximum sequence length of 512 was used and achieved high results. However, when taking a sample of 4000 of the tokenized dataset, the following statistics were found:
                   
              Sample token lengths:
              mean: 404.08275
              median: 196.0
              p80: 555.4
              p90: 831.0
              p99: 3739.7
              max: 13174
              Total samples analyzed: 4000
              Tokens > 256: 41.65%
              Tokens > 512: 21.98%

 Based on this analysis, I decided to test running the model with a maximum token sequence length of 256, which would significantly improve computational time to see if my high accuracy would be maintained. This held true and 256 became the value of this hyperparameter. I hypothesize this is due to how many of the ques that may signify phishing scams appear early in email body text, such as calls to action, prize rewards, or asking for information.


## **Combating DataLeakage**

 In the initial test training run(unfinished_test_Training_DistilBERT), results were as follows:

              Final Validation Metrics
              Accuracy:      0.990867
              Precision:     0.990099
              Recall:        0.990659
              F1:            0.990379

              Confusion Matrix (rows=true, cols=pred)
              Phishing       [3878   35]
              Non-phishing   [  33 3500]
    
 These results seemed suspiciously high. The next step was adding checks for data leakage. 

 - Added removing duplicate emails to my preprocessing steps.
       8027 duplicate emails were removed from the dataset. Before, the same emails were showing up in both the training, validation, and test groups, and the model could have simply been memorizing those answers, inflating the accuracy.


 - Counted the number of URLs to see if the phishing emails all had a significantly higher percentage. This did not seem to be significantly inflating the accuracy of the model. Even if it were, this would be a legitimate check for phishing classification rather than an issue in data processing.

              url_count  email_count    length
              label                                    
              0       1.664196     0.735451  1822.96542
              1       0.416848     0.209896  1091.64203

 - Checked if keywords "phish," "phishing," or "spam" were appearing in the dataset. This was to check if there was potentially extra labeling in the body text sections. These percentages did not seem high enough to significantly inflate the accuracy.

              Percent of emails containing keywords:
              phish    %0.1450
              phishing %0.1302
              spam     %4.9883


 These additional preprocessing steps struck a balance between limited computational resources, checking for and preventing data leakage, and preserving the useful aspects of the data sets.  




## **Model Development and Training**

### **Model Architecture:**

 The model used is **DistilBERT (distilbert-base-uncased)**, a smaller and faster version of BERT that retains much of BERT's language understanding capability while being more computationally efficient. A sequence classification head was added for binary classification. The initial test run with BERT took too long to run, and its runtime session was canceled. So switching to DistilBERT was an optimal choice for the computational power and time available. 

 BERT is well-suited for phishing detection because its bidirectional, contextual language representations allow it to analyze subtle semantic and structural factors that are essential to identifying cues for phishing. BERT is excellent for sentiment analysis and scam detection  
                                                 


 The dataset was split into stratified groups:

               Train:       %80   ~ 59562  emails (subsampled to 30000 for compute time)
               Valiation:   %10   ~ 7446   emails
               Test:        %10   ~ 7446   emails

 Then, to optimize computing time, a randomized subsample of the training data was implemented. I used a randomized 30,000 emails from the train group. This lowered computation time yet preserved the diversity of the dataset. Next steps, when I have access to more computing power, would be to remove this step.


### **Key training choices:**

 - Loss Function: Cross-entropy loss (handled internally by the Hugging Face model)
 - Optimizer: AdamW
 - Learning Rate: 2e-5 (standard learning rate)
 - Batch Size: 32 (good for a max token length of 256)
 - Epochs: 3  (2 - 4 epochs recommended, initial testing with 3 epochs, balanced high accuracy and compute time well)
 - Gradient Clipping: 1.0
 - Early Stopping: Based on validation F1 score to save time training (an extra step to preserve computation)

Epochs, Optimizer, Batch Size, and Learning Rate are all well-established hyperparameters that were recommended by the creators of BERT and worked well for this task to prioritize accuracy and lower computational expense. (Devlin)

After initial test runs with these hyperparameters achieved extraordinarily high accuracy, and after testing for data leakage (through preprocessing) and overfitting (with test data set), adjusting the hyperparameters seemed to be unnecessary for the 8-10 hour timeframe of this project. 


## **Model Evaluation and Results**

         === Best epoch validation statistics ===
              Val Loss:            0.0416
              Val Accuracy:        0.9909
              Val Precision:       0.9901
              Val Recall :         0.9907
              Val F1  :            0.9904
              
              === Best model final report on stratified test data ===
              Test Loss:           0.0387
              Test Accuracy:       0.9915
              Test Precision:      0.9896
              Test Recall :        0.9926
              Test F1  :           0.9911
              
              === Test set confusion matrix ===
               Phishing       [3876   37]
               Non-Phishing   [  26 3507] 
              
              === Test set classification report ===
               precision    recall  f1-score   support
              
               0     0.9933    0.9905    0.9919      3913
               1     0.9896    0.9926    0.9911      3533
              
               accuracy                            0.9915      7446
               macro avg       0.9914    0.9916    0.9915      7446
               weighted avg    0.9915    0.9915    0.9915      7446



These results were excellent. There is also more confidence behind them because of the additional checking for data leakage implemented for the final training. The model also did evenly well across the two evenly balanced classifications. 


- High **accuracy** across both the validation and test datasets means the model classified the vast majority of the emails correctly. 

- This high **precision** indicated that 99% of the emails identified as phishing were actually phishing. 
 Additional metrics on top of accuracy are important to ensure the model is actually performing well on all classes. This is not an enormous concern, as this data set was very well balanced, but it is still important for gaining a well-rounded picture of the results.

- **Recall** indicated that the model accurately caught 99% of the phishing emails. 
 False negatives are more dire in a social situation where a user is trusting their phishing detection tool to warn them of potentially dangerous emails. Having high recall is very important for making sure misleading false negatives are not slipping through.

- **F1** gives a general overview of the model's performance by balancing false positives and false negatives. 
 The F1 score is a very important metric for assessing the quality of phishing detection because both false positives and false negatives are significant.
 
 
 These high results were present in both the validation stage of training the model and during the final testing of the model using the stratified test set. The model is highly effective at identifying phishing emails.



## **Discussion**

### **Fit for the task:**

- This particular dataset was an excellent fit for this task. It was specifically created for phishing detection. It was an extensive dataset with over 80,000 pieces of data compiled from six sources. This variety of input helped ward off overfitting and made the final dataset more robust. The fact that the phishing and non-phishing classes were evenly balanced and labeled consistently also contributed to this dataset being an excellent fit for this task.

- Finetuning a pre-trained DistilBERT model from the Hugging Face transformers library was a great fit for this task. DistilBERT is a smaller version of BERT that requires less computing power yet retains strong contextual language analysis abilities. Its bidirectional representations allow it to capture and recognize the patterns in tone and linguistic cues that phishing emails often contain, like urgency, calls to action, or misspellings. DistilBERT was easily finetuned for the binary classification task of phishing identification. This was also much less time consuming and less expensive then training a model from scratch.

- Using a variety of metrics to analyze the performance of the model to fit the task. Multiple evaluation types, such as Accuracy, Precision, Recall, and F1, were all useful metrics that provided a more accurate and comprehensive overview of the model's performance when combined. Their consistency across the results also brings confidence to the effectiveness of the final model.


### **Social Impact and Broader Implications**
Phishing is a major cybersecurity threat with dire consequences. Phishing can lead to financial loss, identity theft, and even erosion of trust in digital communication. Phishing scams often attempt to target vulnerable populations who may not have the ability to recognize threats as effectively. However, this is also an incredibly pervasive issue with hundreds of thousands of people falling for or being affected by phishing. There is a huge opportunity for social good in combating phishing if people have access to robust tools to confidently identify these threats. 

Phishing detection systems like this one can contribute to:
- Improved email security tools
- Reduced exposure to scams
- Increased digital safety for vulnerable populations


### **Limitations**

 Some limitations may arise from how the dataset was compiled from multiple sources, and it did include exact duplicates. Additionally, many of the emails were stylistically similar and, on average, very short in length, with a median of 735 characters. Because the model performed so exceptionally well, there is a risk of overfitting. Using a completely stratified final test was intended to mitigate and test this; however, it remains a real possibility. Additionally, if a more robust tool were to be developed, I would likely adjust my methodology to minimize the impact of computing time on decision-making. Many of my choices for this project were based on my limited time and computational abilities.

### **Future Improvement**

For future improvement, my next step would be to transfer training and save the model from Google Colab to another environment. This way, the model could be accessed and used beyond the brief allocated runtime slot that the free version of Colab used here. Then I would implement the ability to feed a text file with email body text into the trained model so it could classify specific email examples. This step would make the project and model more applicable to real-world situations. This model would be a tool used to classify and inform users based on individually entered emails. An even further development, if I were to continue this project, would be to connect it to a website or a browser extension that would be accessible to the public. This development would allow more people to have access to the social good of an accurate tool that can identify and warn against dangerous phishing scams in emails. Creating a phishing detection tool that achieves the same results as this model would be an excellent example of social good.


## **Citations**

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019, June 7). Bert: Pre-training of deep bidirectional Transformers for language understanding. ACL Anthology. https://aclanthology.org/N19-1423/ 

###Articles cited at the request of dataset authors:

Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. ArXiv.org. https://arxiv.org/abs/2405.11619


