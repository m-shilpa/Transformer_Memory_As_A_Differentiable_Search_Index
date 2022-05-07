# Transformer Memory As A Differentiable Search Index

Link to the paper: <a href='https://arxiv.org/abs/2202.06991#:~:text=In%20this%20paper%2C%20we%20demonstrate,the%20parameters%20of%20the%20model.'>Paper</a> 

Gist of the paper:
  
  An information retrival(IR) system is one that maps a user query to the relevant documents. The documents are represented by unique identifiers called the document identifiers.
  
  A classic IR system uses the retrieve-then-rank strategy. Retrival of the document id is done using the inverted index i.e, the document ids are mapped to the words, eg. for the word 'Information', all the documents that contain this word will be mapped to it. Similarly, the user's query will be mapped to all the documents that contain the keywords. Next, these documents are ranked using methods like tfid etc. to get the most relevant documents.
  
  In this paper, a transformer is trained to directly provide the document id, given a user query. The idea is to store all the information of the documents in the parameters of the transformer. An analog to this, is a student who studies for exams. The student studies all her textbooks and learns their content. After learning, the student now knows which textbook has what information. During exams, when questions are posed, the student maps the question to the relevent textbook and provides the answer.
  
  The paper uses the below model:
  
  ![image](https://user-images.githubusercontent.com/36926868/167251030-6a5591b1-fc9d-4c4e-a0ab-25ec27133312.png)

  In this model we can see 2 stages:
  1. Indexing stage - where the model learns the content of the document and maps the document to a document id.
  2. Retreval stage - where the document ids are mapped to the query. The output of this stage is the document id that is most relevant to the query.

  The paper also tells about the different stategies that could be used to represent documents and ids.
  
  ## Indexing Stategies:
  1. Input2Target: The document tokens are mapped to the document id.
  2. Target2Input: This is the reverse of Input2Target. The douments are generated from the document ids.
  3. Bidirectional: This is combination of Input2Target and Target2Input.
  4. Span Corruption
  
  ## Docuement Representation Stategies:
  1. Direct Indexing: This strategy takes the first L tokens of the document only. The order of words is preserved.
  2. Set Indexing: This strategy removes any repeated words and stopwords from the document. The resultant document is then passed to the model. A disadvantage of this method could be that the context may be lost when repeated words are removed. 
eg. Sentence: "I was happy since my son was happy".
Here if the second occurance of happy is removed, then the context is lost and is tough to understand the meaning of the sentence.
  3. Inverted Index: This strategy takes a random chunk of the document containing k tokens. This allows looking beyond the L tokens as in the direct indexing. This also could have the disadvantage of losing context if a completly random chunk of the document is picked. This is like watching a random clip of a movie, which may not contain any useful information of the movie itself.

## Representing Docids for Retrival:
1. Unstructured Atomic Identifiers: Each of the documents are assigned arbitrary integer identifiers.
2. Naively Structured Identifiers
3. Semantically Structured Identifiers


The paper's Impementation Details:

* Dataset: Natural Questions(NQ). This dataset contains 307K query-document pairs where the queries are natural language questions and the documents are wikipedia articles.
* Model: T5 model
* Indexing Stategy:

## My Implementation of the paper:
Gist:
1. DataSet Size: 50K documents and 50K questions
2. Model: T5
3. Epochs: 50
4. Loss: 
5. Accuracy on validation dataset :

## Strategy used:
1. Multi-task training using the t5 model was performed on the following tasks:
    1. Indexing : Here, the input is the document and the output is the docid.
    2. Retrival: Here, the input is the question and the output is the docid.
   To differentiate between the two tasks, the task name is appended to the input string. 'Indexing' for the indexing task and 'Retrival' for the retrival task.
2. The initution is to make the model:
    1. Understand the contents of the document. 
    2. Learn the docid of each document.
    3. Understand the relationship between a question and a document that answers the question.

## Data Preprocessing:
  1. The documents in the dataset contained html tags. These were removed using BeatifulSoup.
  2. I used the direct indexing approach for document representation, so i took only the first 1000 words from the document.
  3. I generated the docids by providing a unique interger to each of the documents starting from 1 to n.

## Train-Test Split:
  1. All the documents are provided for training. This is done since it is illogical for model to predict the docid of a document it has never seen before during validation.
  2. Only the questions are split in a ratio of 80:20. 
  
## Batching for training/testing :
  1. Both the indexing and retrival tasks are trained together. 
  2. Since the length of the documents are far larger than that of the questions, if they are batched together, the padding for the questions would increase a lot.To avoid this, each of the tasks are trained in separate batches.


## Loss:













