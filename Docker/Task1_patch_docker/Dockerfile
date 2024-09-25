## Pull from existing image
FROM nvcr.io/nvidia/pytorch:23.10-py3
  
## Copy requirements
COPY ./requirements.txt .

## Install Python packages in Docker image
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# ## Copy all files (here "./src/run_inference.py")
# COPY ./ ./

RUN mkdir /nephrobit/
COPY ./ /nephrobit
RUN chmod -R 777 /nephrobit

## Execute the inference command 
CMD ["/nephrobit/src/inference_patch.py"]
ENTRYPOINT ["python3"]