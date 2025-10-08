1. Dataset Download (AG News)
  You load the AG News dataset from the Hugging Face datasets library.
  Each record contains:
    text: The content of the article (title + description)
    label: A numerical category (0 = World, 1 = Sports, etc.)
  So, you now have thousands of labeled text samples for training and evaluation.
2. Text Preprocessing + TF-IDF Vectorization
  Raw text cannot be directly used by most ML algorithms — they only work with numbers.
  So we convert the text into a numerical representation using TF-IDF (Term Frequency–Inverse Document Frequency).
  TF-IDF measures how important a word is in a document relative to the entire dataset.
  Common words like “the” or “is” get low weight.
  More informative words like “football” or “earnings” get higher weight.
  This produces a sparse matrix (like a big spreadsheet) where:
    Each row = one article
    Each column = a unique word
    Each value = how important that word is for that article
3. Model Training (Logistic Regression)
  The TF-IDF matrix is then used as input to a Logistic Regression classifier.
  Logistic Regression is a linear model that tries to draw boundaries between classes.
  Even though it’s simple, it performs surprisingly well for text classification, especially with TF-IDF.
  The model learns patterns like:
    If “goal”, “match”, “team” appear often → likely “Sports”
    If “stocks”, “market”, “CEO” appear → likely “Business”
4. Evaluation
  Once trained, the model is tested on unseen data (the test split of AG News).
  It predicts labels for each test article.
  You measure accuracy and sometimes more detailed metrics (precision, recall, F1).
5. Purpose
  The project demonstrates the classic NLP pipeline before deep learning:
  Text preprocessing
  Feature extraction (TF-IDF)
  Supervised model training (Logistic Regression)
  It’s a baseline model for text classification — something data scientists build to:
  Compare against deep learning methods (like BERT)
  Prototype quickly
  Understand dataset characteristics
  Deploy lightweight models that don’t need GPUs
