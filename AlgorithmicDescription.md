# 🧠 TF-IDF Vectorization and Logistic Regression

This section provides both a **mathematical** and **intuitive** explanation of two fundamental algorithms used in Natural Language Processing (NLP):  
**TF-IDF Vectorization** and **Logistic Regression**.

---

## 📘 1. TF-IDF Vectorization (Term Frequency–Inverse Document Frequency)

The TF-IDF (Term Frequency–Inverse Document Frequency) score for a term *t* in a document *d* within a corpus *D* is calculated as the product of its **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**.

### **Given:**
- A corpus  
  $$D = \{ d_1, d_2, \ldots, d_N \}$$  
- A vocabulary of terms  
  $$T = \{ t_1, t_2, \ldots, t_M \}$$

---

### **1. Term Frequency (TF)**

The **Term Frequency** measures how frequently a term appears in a document.  
A common way to calculate it is:

$$
TF(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
$$

This gives higher weight to words that appear often in the same document.

---

### **2. Inverse Document Frequency (IDF)**

The **Inverse Document Frequency** measures how important a term is across the entire corpus.  
It *down-weights* common terms and *up-weights* rare ones:

$$
IDF(t, D) = \log\left(\frac{\text{Total number of documents in corpus } D}{1 + \text{Number of documents in } D \text{ containing term } t}\right)
$$

The "+1" in the denominator avoids division by zero when a term appears in every document.

---

### **3. TF-IDF Score**

Finally, the **TF-IDF** score is computed as the product of TF and IDF:

$$
TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)
$$

Each document *d* can then be represented as a vector of these weights:

$$
\vec{v_d} = [TF\text{-}IDF(t_1, d), TF\text{-}IDF(t_2, d), \ldots, TF\text{-}IDF(t_M, d)]
$$

---

### **Layman’s Explanation**

Think of TF-IDF as a way to measure **how important each word is in a text**:
- Words that appear often in one document (like “data” in a research article) get a **high score**.
- Words that appear everywhere (like “the” or “and”) get a **low score**.

In simple terms:
- **TF** tells you how frequent a word is in that document.
- **IDF** tells you how unique it is across all documents.
- The product (TF × IDF) gives a balanced measure of importance.

So, TF-IDF turns sentences or entire documents into **vectors of meaningful numbers**, allowing algorithms like **Logistic Regression** to work with text.

---

## 📗 2. Logistic Regression

### **Mathematical Description**

Logistic Regression is a **supervised learning algorithm** used for **binary classification** — deciding between two classes (e.g., spam vs. not spam).

Given:
- A dataset of input vectors $$\( X = \{ x_1, x_2, \ldots, x_n \} \)$$
- Corresponding binary labels $$\( y_i \in \{0, 1\} \)$$

The logistic model computes the probability that a sample belongs to class 1:

$$
P(y=1|x) = \sigma(w^T x + b)
$$

where:
- \( w \) = weight vector (learned parameters)  
- \( b \) = bias term  
- $\( \sigma(z) \)$ = **sigmoid function** defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The model predicts:

$$
\hat{y} =
\begin{cases}
1, & \text{if } P(y=1|x) \geq 0.5 \\
0, & \text{otherwise}
\end{cases}
$$

**Training Objective:**
The parameters $\( w \)$ and $\( b \)$ are learned by minimizing the **log-loss (binary cross-entropy)**:

$$
L(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y_i}) + (1 - y_i) \log(1 - \hat{y_i}) \right]
$$

---

### **Layman’s Explanation**

Imagine you’re building a spam detector. Logistic Regression looks at features (like TF-IDF scores of words) and learns how each feature contributes to the **probability** that an email is spam.

- It doesn’t output just “spam” or “not spam” — it outputs a **probability** between 0 and 1.  
- The **sigmoid function** turns any input into that range.  
  For example, if the model computes a score of 2.0, the sigmoid converts it into about 0.88 → meaning “88% chance of being spam.”

During training, the algorithm **adjusts the weights** for each feature so that predictions match real outcomes as closely as possible.

Think of it like adjusting knobs on a control board until the model’s guesses line up with reality.

---

## 🧩 Connecting TF-IDF and Logistic Regression

When combined:
1. **TF-IDF** turns words into numerical features.  
2. **Logistic Regression** uses those numbers to classify the text (e.g., positive/negative sentiment, spam/ham, etc.).

### Example Workflow
1. Take raw text → “This movie was fantastic and inspiring.”
2. Use TF-IDF → Convert it into a numeric vector showing how important each word is.
3. Feed the vector into Logistic Regression → Output a probability (e.g., 0.93 positive sentiment).

---

## 🧠 Summary in Simple Terms

- **TF-IDF** converts text into meaningful numbers — telling us which words are significant for each document.
- **Logistic Regression** learns how to use those numbers to make predictions — like classifying whether something belongs to one group or another.

Together, they form a simple yet powerful foundation for many NLP systems such as:
- Spam detection
- Sentiment analysis
- Topic classification

---
