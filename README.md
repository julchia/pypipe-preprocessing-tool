# Spanish Text Preprocessing Tool

### 🛠️ Work in progress 🛠️

## In progess:

- Vocabulary creator
- Word2Vec featurizer
- Run bash commands
- Write a real readme 😅

## About the project

<p align='justify'>The Spanish Text Preprocessing Tool is designed to facilitate the preprocessing of text data through the execution of isolated or sequential processes.</p>

<p align='justify'>The project was born as a personal hobby that I use to learn and have fun in my free time, along with the need to combine processes and tools into a single service that I use daily in other personal projects.</p>

<p align='justify'>While the project is not yet in a production stage, it already offers some functionality, such as normalization via regex and vectorization via a vectorizer.</p>

<p align='justify'>In short, the Spanish Text Preprocessing Tool is a service that allows users to manage and dynamically create different pipelines for text data preprocessing simply by setting a configuration inside a static file (currently located in src/configs/preprocessing_config.json). This static file triggers the creation of a pipeline that allows the processes that appear in the configuration to be used individually or sequentially (although sequential functionality is not yet implemented). It is worth mentioning that each process can work as an isolated object, allowing them to work in a decoupled way from the configuration.</p>

<p align='justify'>At present, I haven't yet focused on developing commands that automate flows. However, I'll provide a simple example of a possible use of the tool:</p>

Suppose we have the following raw spanish corpus:

```
corpus = [
    "HOLA!!1111 gente lindaaaaa!!!",
    "el nombre essssss @Pedro re loco jjajajajjja",
    "mi correo es pedrito@gmail.com",
    "su página es www.pedrito.com ...."
]
```
We can create en new pipeline object:

```
pipe_1 = Pipeline(
    pipeline_conf=PREPROCESSING_CONFIG
)
```
And just use the regex normalizer setted in ```src/configs/preprocessing_config.json``` over the corpus:

```
norm_corpus = pipe_1.regex_normalization.normalize_text(corpus)
```
We get the following output

```
['hola gente linda', 'el nombre es MENTION muy loco jaja', 'mi correo es EMAIL', 'su pagina es URL']
```

Then, using the same pipeline object, we can train the sklearn count vectorizer and vectorize an unseen corpus. Similar to the regex normalizer, the vectorizer was set in the configuration file to configure the vectorizer:

```
pipe_1.count_vect_featurizer.train(norm_corpus)

unseen_corpus = [
    "hola gente",
    "jaja su nombre es pedro",
    "el correo de pedro es MAIL"
]

vectors = pipe_1.count_vect_featurizer.process(unseen_corpus)
```

We get the following output:

```
[[0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0]
 [1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0]]
```
