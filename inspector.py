from datetime import datetime
from time import time

import mxnet as mx


def available_gpu_devices():
    try:
        mx.nd.array([1, 2, 3], ctx=mx.gpu())
    except mx.MXNetError:
        return False
    return True


def perform_matrix_operations(device):
    # create a matrix of [2, 3] shape
    print("Matrix A")
    a = mx.nd.array([[1.0,2.0,3.0], [4.0,5.0,6.0]], ctx=device)
    print(a)
    # create a matrix of [3, 2] shape
    print("\nMatrix B")
    b = mx.nd.array([[1.0,2.0], [3.0, 4.0], [5.0,6.0]], ctx=device)
    print(b)
    # perform a matrix product of those two tensors
    print("\nMatrix Product of Matrices A & B")
    start_time = time()
    c = mx.nd.dot(a,b)
    end_time = time()
    elapsed_time = end_time - start_time
    print(c)
    print("... time to finish the workload: %f s" % elapsed_time)
    print("\n... matrix operations has been completed")


def test_cpu_gpu_computations():
    # Perform CPU computations
    print("\n++++++++++++++++++++++++++++++")
    print("... placing matrix operations on available CPU devices")
    cpu_device = mx.cpu()
    perform_matrix_operations(cpu_device)
    print("++++++++++++++++++++++++++++++")

    # Perform GPU computations
    print("\n++++++++++++++++++++++++++++++")
    # check if the GPU libraries are present
    if available_gpu_devices():
        print("... placing matrix operations on available GPU devices")
        gpu_device = mx.gpu()
        perform_matrix_operations(gpu_device)
    else:
        print("... GPU devices or libraries are not present!")
    print("++++++++++++++++++++++++++++++")


def verify_mxnet_installation():
    print("\n++++++++++++++++++++++++++++++")
    # verification of simple MXNet installation
    print("... verifying basic MXNet installation")
    # - create a simple matrix
    try:
        print("\n...initialize matrix of shape (2,3) with ones")
        a = mx.nd.ones((2, 3))
        print(a)
        print("\n...multiply the individual items by value of 2 & add value of 1")
        b = a * 2 + 1
        print(b)
        print("\nA simple verification of MXNet installation has been successful!")
    except:
        print("A simple verification of MXNet installation has failed!")
    print("++++++++++++++++++++++++++++++")


def main():
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+++ Simple CPU/GPU Computation Test with MXNet +++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("start: " + str(datetime.now()))
    # verify MXNet installation
    verify_mxnet_installation()
    # test CPU/GPU computations
    test_cpu_gpu_computations()
    print("\nend: " + str(datetime.now()))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    main()
