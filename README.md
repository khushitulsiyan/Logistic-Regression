# Logistic-Regression

Logistic Regression is a variation of Linear Regression, useful when the observed dependent variable, y, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables.

Logistic regression fits a special s-shaped curve by taking the linear regression function and transforming the numeric estimate into a probability with the following function, which is called the sigmoid function 𝜎:

ℎ_𝜃(𝑥)=𝜎(𝜃𝑇𝑋)=𝑒(𝜃_0+𝜃_1𝑥_1+𝜃_2𝑥_2+...)1+𝑒(𝜃_0+𝜃_1𝑥_1+𝜃_2𝑥_2+⋯)
 
Or:
𝑃𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦𝑂𝑓𝑎𝐶𝑙𝑎𝑠𝑠_1=𝑃(𝑌=1|𝑋)=𝜎(𝜃𝑇𝑋)=𝑒𝜃𝑇𝑋1+𝑒𝜃𝑇𝑋
 
In this equation,  𝜃𝑇𝑋  is the regression result (the sum of the variables weighted by the coefficients), exp is the exponential function and  𝜎(𝜃𝑇𝑋)  is the sigmoid or logistic function, also called logistic curve. It is a common "S" shape (sigmoid curve).

So, briefly, Logistic Regression passes the input through the logistic/sigmoid but then treats the result as a probability:



The objective of the Logistic Regression algorithm, is to find the best parameters θ, for  ℎ_𝜃(𝑥)  =  𝜎(𝜃𝑇𝑋) , in such a way that the model best predicts the class of each case.
