#Text Preprocessing Tool

### 🛠️ Work in progress 🛠️

## In progess:

- Unit tests
- Build package
- Cli commands
- Write a real readme 😅

## About the project

<p align='justify'>The Text Preprocessing Tool is designed to facilitate the preprocessing of text data through the execution of isolated or sequential processes. At the moment, designed mainly to work with text in Spanish</p>

<p align='justify'>The project was born as a personal hobby that I use to learn and have fun in my free time, along with the need to combine processes and tools into a single service that I use daily in other personal projects.</p>

<p align='justify'>In short, the Text Preprocessing Tool is a service that allows users to manage and dynamically create different pipelines for text data preprocessing simply by setting a configuration inside a static file (src/configs/preprocessing.json). This static file triggers the creation of a pipeline that allows the processes that appear in the configuration to be used individually or sequentially (although sequential functionality is not yet implemented). It is worth mentioning that each process can work as an isolated object, allowing them to work in a decoupled way from the configuration.</p>

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
We can create a new pipeline object, specifying the path to the configurations:

```
pipe_1 = Pipeline(
    config_path=paths.PREPROCESSING_CONFIG_PATH
).create_pipeline()
```
And simply use the regex normalizer that is set in configurations to normalize the corpus:

```
norm_corpus = pipe_1.regex_norm.normalize_text(corpus)

```
We get the following output

```
['hola gente linda', 'el nombre es MENTION muy loco jaja', 'mi correo es EMAIL', 'su pagina es URL']
```

Then, using the same pipeline object, we can train the sklearn count vectorizer (or any other vectorizer that is in the settings) and vectorize an unseen corpus. Similar to the regex normalizer, the vectorizer was set in the configuration file to configure the vectorizer:

```
pipe_1.countvec.train(norm_corpus)

unseen_corpus = [
    "hola gente",
    "jaja su nombre es pedro",
    "el correo de pedro es MAIL"
]

vectors = pipe_1.countvec.process(unseen_corpus)

```

We get the following output:

```
[[0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0]
 [1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0]]
```

At the same time, if we have created a Pipeline object, we can also execute the processes sequentially

```
pipe_1.process_corpus_sequentially(corpus=corpus)
```

Handler objects (src/core/pipeline/handlers.py) offer a quick way to set the actions that each process can execute during its sequential execution.
