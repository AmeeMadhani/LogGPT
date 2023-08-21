# logGPT

This project was inspired by the original [privateGPT](https://github.com/imartinez/privateGPT). Most of the description here is inspired by the original privateGPT.

In this model, We have replaced the GPT4ALL model with Vicuna-7B model and we are using the InstructorEmbeddings instead of LlamaEmbeddings as used in the original privateGPT. Both Embeddings as well as LLM will run on GPU instead of CPU. It also has CPU support if you do not have a GPU (see below for instruction).

Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves your execution environment at any point. You can ingest documents and ask questions on policy infrigement without an internet connection!

Built with [LangChain](https://github.com/hwchase17/langchain) and [Vicuna-7B](https://huggingface.co/TheBloke/vicuna-7B-1.1-HF) and [InstructorEmbeddings](https://instructor-embedding.github.io/)

# Environment Setup

Install conda

```shell
conda create -n localGPT
```

Activate

```shell
conda activate localGPT
```

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```


## Instructions for ingesting your own dataset

Put any and all of your .txt, .pdf, or .csv files into the SOURCE_DOCUMENTS directory
in the load_documents() function, replace the docs_path with the absolute path of your source_documents directory.

The current default file types are .txt, .pdf, .csv, and .xlsx, if you want to use any other file type, you will need to convert it to one of the default file types.

Run the following command to ingest all the data.

`defaults to cuda`

```shell
python ingest.py
```

It will create an index containing the local vectorstore. Will take time, depending on the size of your documents.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `index`.

Note: When you run this for the first time, it will download take time as it has to download the embedding model. In the subseqeunt runs, no data will leave your local enviroment and can be run without internet connection.

## Ask questions to your documents, locally!

In order to ask a question, run a command like:

```shell
python run_localGPT.py
```

And wait for the script to require your input.

```shell
> Enter a query:
```

Hit enter. Wait while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Note: When you run this for the first time, it will need internet connection to download the vicuna-7B model. After that you can turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

# Run it on CPU

By default, localGPT will use your GPU to run both the `ingest.py` and `run_localGPT.py` scripts. But if you do not have a GPU and want to run this on CPU, now you can do that (Warning: Its going to be slow!). You will need to use `--device_type cpu`flag with both scripts.

For Ingestion run the following:

```shell
python ingest.py --device_type cpu
```

In order to ask a question, run a command like:

```shell
python run_localGPT.py --device_type cpu
```

# Run quantized for M1/M2:

GGML quantized models for Apple Silicon (M1/M2) are supported through the llama-cpp library, [example](https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML). GPTQ quantized models that leverage auto-gptq will not work, [see here](https://github.com/PanQiWei/AutoGPTQ/issues/133#issuecomment-1575002893). GGML models will work for CPU or MPS.

## Troubleshooting

**Install MPS:**
1- Follow this [page](https://developer.apple.com/metal/pytorch/) to build up PyTorch with Metal Performance Shaders (MPS) support. PyTorch uses the new MPS backend for GPU training acceleration. It is good practice to verify mps support using a simple Python script as mentioned in the provided link.

2- By following the page, here is an example of what you may initiate in your terminal

```shell
xcode-select --install
conda install pytorch torchvision torchaudio -c pytorch-nightly
pip install chardet
pip install cchardet
pip uninstall charset_normalizer
pip install charset_normalizer
pip install pdfminer.six
pip install xformers
```

# How does it work?

Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `InstructorEmbeddings`. It then stores the result in a local vector database using `Chroma` vector store.
- `run_localGPT.py` uses a local LLM (Vicuna-7B in this case) to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.
- You can replace this local LLM with any other LLM from the HuggingFace. Make sure whatever LLM you select is in the HF format.

# How to select different LLM models?

The following will provide instructions on how you can select a different LLM model to create your response:

1. Open up `run_localGPT.py`
2. Go to `def main(device_type, show_sources)`
3. Go to the comment where it says `# load the LLM for generating Natural Language responses`
4. Below it, it details a bunch of examples on models from HuggingFace that have already been tested to be run with the original trained model (ending with HF or have a .bin in its "Files and versions"), and quantized models (ending with GPTQ or have a .no-act-order or .safetensors in its "Files and versions").
5. For models that end with HF or have a .bin inside its "Files and versions" on its HuggingFace page.

   - Make sure you have a model_id selected. For example -> `model_id = "TheBloke/guanaco-7B-HF"`
   - If you go to its HuggingFace [repo](https://huggingface.co/TheBloke/guanaco-7B-HF) and go to "Files and versions" you will notice model files that end with a .bin extension.
   - Any model files that contain .bin extensions will be run with the following code where the `# load the LLM for generating Natural Language responses` comment is found.
   - `model_id = "TheBloke/guanaco-7B-HF"`

     `llm = load_model(device_type, model_id=model_id)`

6. For models that contain GPTQ in its name and or have a .no-act-order or .safetensors extension inside its "Files and versions on its HuggingFace page.

   - Make sure you have a model_id selected. For example -> model_id = `"TheBloke/wizardLM-7B-GPTQ"`
   - You will also need its model basename file selected. For example -> `model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"`
   - If you go to its HuggingFace [repo](https://huggingface.co/TheBloke/wizardLM-7B-GPTQ) and go to "Files and versions" you will notice a model file that ends with a .safetensors extension.
   - Any model files that contain no-act-order or .safetensors extensions will be run with the following code where the `# load the LLM for generating Natural Language responses` comment is found.
   - `model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"`

     `model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"`

     `llm = load_model(device_type, model_id=model_id, model_basename = model_basename)`

7. Comment out all other instances of `model_id="other model names"`, `model_basename=other base model names`, and `llm = load_model(args*)`

# System Requirements

## Python Version

To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11

To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   - Universal Windows Platform development
   - C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the "gcc" component.
