# import os
# os.environ['GRPC_TRACE'] = 'all'
# os.environ['GRPC_VERBOSITY'] = 'DEBUG'

import grpc
import tensorflow as tf
from fastapi import FastAPI

from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


app = FastAPI()

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

@app.get('/')
async def root():
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'half_plus_two'
    request.model_spec.signature_name = 'serving_default'
    request.model_spec.version.value = 123
    request.inputs['x'].CopyFrom(tf.make_tensor_proto(2., dtype=types_pb2.DT_FLOAT))
    request.output_filter.append('y')
    result = stub.Predict(request, 10.0)
    y = result.outputs['y'].float_val[0]
    return {'res': y}
