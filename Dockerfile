FROM public.ecr.aws/lambda/python:3.9

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN  python -m pip install --upgrade pip
RUN  yum install gcc -y
RUN  pip --no-cache-dir install numpy==1.19.2 scikit-learn==0.20.3
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
#ENV SURPRISE_DATA_FOLDER=/tmp/.surprise_data

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.lambda_handler" ] 
