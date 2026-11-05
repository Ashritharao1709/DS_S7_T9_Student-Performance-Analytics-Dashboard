
# Random Bernoulli Samples
import numpy as np
# Probability of success
p = 0.7
# Generate 10 Bernoulli random samples
samples = np.random.binomial(n=1, p=p, size=10)
print("Bernoulli Samples:")
print(samples)

# Bernoulli Probability Calculation
# Probability of success
p = 0.7
# Probability of failure
q = 1 - p
print("Probability of Success (P(X=1)) =", p)
print("Probability of Failure (P(X=0)) =", q)
# Calculate probability based on user input
x = int(input("Enter outcome (0 for Failure, 1 for Success): "))
if x == 1:
    print("Bernoulli Probability =", p)
elif x == 0:
    print("Bernoulli Probability =", q)
else:
    print("Invalid input! Enter only 0 or 1.")

# Probability Distributions using Python
# 1. Bernoulli Distribution
# 2. Binomial Distribution
# Application: Student Exam Pass/Fail Analysis
import numpy as np
from scipy.stats import bernoulli, binom

# BERNOULLI DISTRIBUTION
print("----- Bernoulli Distribution -----")
# Probability of passing an exam
p = 0.7
# Generate 10 random student outcomes
# 1 = Pass, 0 = Fail
bernoulli_samples = bernoulli.rvs(p, size=10)
print("Probability of Passing:", p)
print("Student Outcomes (1=Pass, 0=Fail):")
print(bernoulli_samples)
# Probability of success and failure
print("\nProbability of Pass:")
print(bernoulli.pmf(1, p))
print("Probability of Fail:")
print(bernoulli.pmf(0, p))
# -----------------------------
# BINOMIAL DISTRIBUTION
# -----------------------------
print("\n\n----- Binomial Distribution -----")
# Number of students
n = 10

# Probability of passing
p = 0.7
# Probability that exactly 8 students pass
k = 8
binomial_probability = binom.pmf(k, n, p)
print("Number of Students:", n)
print("Probability of Passing:", p)
print(f"\nProbability that exactly {k} students pass:")
print(binomial_probability)
# Generate random experiment
random_binomial = binom.rvs(n, p, size=5)
print("\nRandom Binomial Outcomes:")
print(random_binomial)

# Statistical Hypothesis Testing
# 1. Z-Test
# 2. Independent t-Test
# Application: Student Performance Analysis
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import ztest

# Z-TEST
print("----- Z-Test -----")
# Sample marks of students
sample_marks = np.array([78, 82, 85, 88, 90, 76, 95, 89, 84, 91])
# Population mean
population_mean = 80
# Perform one-sample Z-test
z_stat, p_value = ztest(sample_marks, value=population_mean)
print("Sample Marks:")
print(sample_marks)
print("\nPopulation Mean:", population_mean)
print("\nZ-Statistic:", z_stat)
print("P-Value:", p_value)
# Decision
alpha = 0.05
if p_value < alpha:
    print("\nResult: Reject Null Hypothesis")
else:
    print("\nResult: Fail to Reject Null Hypothesis")
# INDEPENDENT t-TEST
print("\n\n----- Independent t-Test -----")
# Marks of two different classes
class_A = np.array([78, 82, 85, 88, 90])
class_B = np.array([72, 75, 70, 68, 74])
# Perform independent t-test
t_stat, p_value = stats.ttest_ind(class_A, class_B)
print("Class A Marks:")
print(class_A)
print("\nClass B Marks:")
print(class_B)
print("\nt-Statistic:", t_stat)
print("P-Value:", p_value)
# Decision
if p_value < alpha:
    print("\nResult: Reject Null Hypothesis")
else:
    print("\nResult: Fail to Reject Null Hypothesis")