# Pypipe: Text Preprocessing Tool

### 🛠️ Work in progress 🛠️

## In progess:

- Unit tests
- Build package
- Add details to the readme file

## About the project

<p align='justify'>Pypipe is a text preprocessing tool designed to facilitate the preprocessing of text data through the execution of isolated or sequential processes. Pypipe is designed to function as a base architecture that decouples text normalization and preprocessing processes from any NLP context.</p>

<p align='justify'>The project was born as a hobby, so don't expect constant regularity! 😅 However, it is also true that Pypipe provides a solution for certain needs in some of my other projects.</p>

## About Pypipe

<p align='justify'>Pypipe is an NLP tool that enables users to manage and create different pipelines for text data preprocessing dynamically. By setting a configuration inside a static file (pypipe/configs/preprocessing.json), users can trigger the creation of a pipeline that allows the processes in the configuration to be used individually or sequentially. Notably, each process can work as an isolated object, allowing for decoupled operation from the configuration.</p>

<p align='justify'>You can take Pypipe base architecture to remove or add additional processes, or simply take Pypipe so you don't have to start a normalization or preprocessing job from scratch.</p>

## Some uses

<p align='justify'>From the command line, it is possible to both execute a pipeline sequentially or create and execute isolated processes. Assuming that the configuration corresponding to the alias 'prepro_1' was set, you can:</p>

- Execute a pipeline sequentially and store it results:

```
python -m pypipe prepro_1 --data path/to/corpus.txt --store 
```
- Create a decoupled process set in the configurations, run it, and save its result:

```
python -m pypipe prepro_1 --data path/to/corpus.txt --process regex_norm --method normalize_text --store
```
<p align='justify'>Without using the command line, we can implement processes in an isolated way without necessarily depending on a pipeline. Each processor is designed to work with generators and process the corpus lazily for better memory management.</p>

Let's suppose you want to use the regex normalizer:

```
config = RegexNormalizer.get_default_configs()
regex_norm = RegexNormalizer.get_isolated_process(config)

data = [
    "HOLA!!1111 gente lindaaaaa!!!",
    "el nombre essssss @Pedro re loco jjajajajjja",
    "mi correo es pedrito@gmail.com",
    "su página es www.pedrito.com ...."
]

regex_norm.normalize_text(data, persist=True)
```
Output:

```
hola gente linda
el nombre es MENTION muy loco jaja
mi correo es EMAIL
su pagina es URL
```
