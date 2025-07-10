# Student Placement Prediction

## Project Overview
This project predicts student placement outcomes using machine learning algorithms based on academic performance, skills, and experience data. By analyzing a dataset of 10,000 students, we identify key factors influencing job placements and compare multiple classification approaches to find the most effective prediction model.

## Dataset
The dataset contains information about 10,000 students with the following features:
- College_ID: Unique identifier for each student
- IQ: Intelligence quotient score
- Prev_Sem_Result: Previous semester academic result
- CGPA: Cumulative Grade Point Average
- Academic_Performance: Overall academic performance score
- Internship_Experience: Whether the student has internship experience (Yes/No)
- Extra_Curricular_Score: Score for participation in extra-curricular activities
- Communication_Skills: Score for communication abilities
- Projects_Completed: Number of projects completed
- Placement: Target variable indicating whether the student was placed (Yes/No)

## Methodology

### Data Preprocessing
- Converted categorical variables (Yes/No) to binary (1/0)
- Analyzed feature distributions and correlations
- Created interaction terms between important features
- Split data into training (80%) and testing (20%) sets

### Exploratory Data Analysis
- Visualized feature distributions using histograms and box plots
- Created correlation heatmaps to identify key predictors
- Analyzed the relationship between internship experience and placement outcomes

### Feature Engineering
- Generated interaction features between important variables like IQ, CGPA, and Academic Performance
- Evaluated the impact of feature engineering on model performance

### Models Implemented
1. **Naive Bayes Classifier**
   - Trained on original features (90.35% accuracy)
   - Trained on enhanced features (87.35% accuracy)

2. **Artificial Neural Network**
   - Architecture: 2 hidden layers (64 and 32 neurons) with dropout regularization
   - Activation: ReLU for hidden layers, Sigmoid for output layer
   - Optimizer: Adam with learning rate of 0.001
   - Achieved 99.65% accuracy on the test set

## Results

### Model Performance Comparison
| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|-------|----------|---------------------|------------------|---------------------|
| Naive Bayes (Original) | 90.35% | 0.76 | 0.61 | 0.68 |
| Naive Bayes (Enhanced) | 87.35% | 0.61 | 0.68 | 0.64 |
| Neural Network | 99.65% | 0.98 | 1.00 | 0.99 |

### Key Findings
- The neural network significantly outperformed the Naive Bayes classifier
- Academic performance metrics (CGPA, Academic_Performance) were the strongest predictors
- Internship experience had a substantial positive impact on placement outcomes
- Adding interaction features did not improve the Naive Bayes model performance, suggesting the relationships are more complex than what Naive Bayes can capture

## Visualizations
The project includes several visualizations:
- Distribution of IQ scores
- Internship experience distribution
- Correlation heatmap with placement
- Training and validation curves for the neural network
- Feature importance analysis
- Confusion matrices for model comparison

## Conclusion
This project demonstrates the effectiveness of deep learning for student placement prediction. The neural network achieved near-perfect classification accuracy by capturing complex relationships between features. The results highlight the importance of academic performance, practical experience, and soft skills in determining placement outcomes.

## Future Work
- Implement more models (Random Forest, XGBoost, SVM)
- Perform hyperparameter tuning using grid search or Bayesian optimization
- Collect additional features like specialized skills or certifications
- Develop an interpretable model for providing actionable feedback to students

## Requirements
- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- TensorFlow 2.x

## How to Run
1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook student_placement_prediction.ipynb`

## License
[MIT License](LICENSE)

---

*This project was created for educational purposes and to demonstrate machine learning applications in educational analytics.*
