## Pull from existing image
FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt-get update && apt-get install --no-install-recommends -y \
    libopenjp2-7-dev libopenjp2-tools \
    openslide-tools \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

## Copy requirements
COPY ./requirements.txt .

## Install Python packages in Docker image
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# ## Copy all files (here "./src/run_inference.py")
# COPY ./ ./

# RUN mkdir /nephrobit/
# COPY ./ /nephrobit
# RUN chmod -R 777 /nephrobit

## Execute the inference command 
CMD ["/nephrobit/src/inference_slide.py"]
ENTRYPOINT ["python3"]