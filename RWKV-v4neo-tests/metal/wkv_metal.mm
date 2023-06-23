#include <torch/extension.h>
#include "wkv_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

torch::Tensor dispatchBackwardKernel(int64_t T_max, int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
                                     torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of elements within the input tensor.
        int numThreads = B * C;

        // Load the custom kernel.
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string kernel_name = std::string("backward_kernel_") + (w.scalar_type() == torch::kFloat ? "float" : "half");
        id<MTLFunction> backwardFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(backwardFunction, "Failed to create function state object for ", kernel_name.c_str());

        // Create a compute pipeline state object for the kernel.
        id<MTLComputePipelineState> backwardPSO = [device newComputePipelineStateWithFunction:backwardFunction error:&error];
        TORCH_CHECK(backwardPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Assuming you have a reference to the device
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();

            // Define the size of the buffer
            NSUInteger bufferSize = sizeof(int) * numThreads; // dataSize is the number of elements you want in the buffer

            // Create the buffer
            id<MTLBuffer> errorBuffer = [device newBufferWithLength:bufferSize options:MTLResourceOptionCPUCacheModeDefault];

            // Set a completion handler for the command buffer
            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> commandBuffer) {
                // After this point, all the commands in the command buffer have finished execution
                // Now we can safely check for logs
                int* errorBufferData = (int*)[errorBuffer contents];
                for(NSUInteger i = 0; i < numThreads; ++i) {
                    if (errorBufferData[i] == 1 ) {
                        NSLog(@"Error in thread %lu -> tid is invalid", (unsigned long)i);
                    }
                    if (errorBufferData[i] == 2) {
                        NSLog(@"Error in thread %lu -> provided context length exceeds buffer size of 1024", (unsigned long)i);
                    }
                }
            }];

            // Encode the pipeline state object and its parameters.
            [computeEncoder setComputePipelineState:backwardPSO];
            [computeEncoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(u) offset:u.storage_offset() * u.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:2];
            [computeEncoder setBuffer:getMTLBufferStorage(v) offset:v.storage_offset() * v.element_size() atIndex:3];
            [computeEncoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:4];
            [computeEncoder setBuffer:getMTLBufferStorage(gy) offset:gy.storage_offset() * gy.element_size() atIndex:5];
            [computeEncoder setBuffer:getMTLBufferStorage(gw) offset:gw.storage_offset() * gw.element_size() atIndex:6];
            [computeEncoder setBuffer:getMTLBufferStorage(gu) offset:gu.storage_offset() * gu.element_size() atIndex:7];
            [computeEncoder setBuffer:getMTLBufferStorage(gk) offset:gk.storage_offset() * gk.element_size() atIndex:8];
            [computeEncoder setBuffer:getMTLBufferStorage(gv) offset:gv.storage_offset() * gv.element_size() atIndex:9];
            [computeEncoder setBytes:&T_max length:sizeof(int64_t) atIndex:10];
            [computeEncoder setBytes:&B length:sizeof(int64_t) atIndex:11];
            [computeEncoder setBytes:&T length:sizeof(int64_t) atIndex:12];
            [computeEncoder setBytes:&C length:sizeof(int64_t) atIndex:13];
            [computeEncoder setBuffer:errorBuffer offset:0 atIndex:14];

            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadGroupSize = backwardPSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > numThreads) {
                threadGroupSize = numThreads;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }
    return y;
}

torch::Tensor dispatchForwardKernel(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of elements within the input tensor.
        int numThreads = B * C;

        // Load the custom kernel.
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string kernel_name = std::string("forward_kernel_") + (w.scalar_type() == torch::kFloat ? "float" : "half");
        id<MTLFunction> forwardFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(forwardFunction, "Failed to create function state object for ", kernel_name.c_str());

        // Create a compute pipeline state object for the kernel.
        id<MTLComputePipelineState> forwardPSO = [device newComputePipelineStateWithFunction:forwardFunction error:&error];
        TORCH_CHECK(forwardPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Encode the pipeline state object and its parameters.
            [computeEncoder setComputePipelineState:forwardPSO];
            [computeEncoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(u) offset:u.storage_offset() * u.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:2];
            [computeEncoder setBuffer:getMTLBufferStorage(v) offset:v.storage_offset() * v.element_size() atIndex:3];
            [computeEncoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:4];
            [computeEncoder setBytes:&B length:sizeof(int64_t) atIndex:5];
            [computeEncoder setBytes:&T length:sizeof(int64_t) atIndex:6];
            [computeEncoder setBytes:&C length:sizeof(int64_t) atIndex:7];

            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadGroupSize = forwardPSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > numThreads) {
                threadGroupSize = numThreads;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }

    return y;
}

torch::Tensor mps_forward_kernel(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    // Check whether the input tensors reside on the MPS device and whether they are contiguous.
    TORCH_CHECK(w.device().is_mps() && u.device().is_mps() && k.device().is_mps() && v.device().is_mps() && y.device().is_mps(),
                "All tensors must be MPS tensors");
    TORCH_CHECK(w.is_contiguous() && u.is_contiguous() && k.is_contiguous() && v.is_contiguous() && y.is_contiguous(),
                "All tensors must be contiguous");

    // Check the supported data types for the forward kernel.
    TORCH_CHECK(w.scalar_type() == torch::kFloat || w.scalar_type() == torch::kHalf,
                "Unsupported data type: ", w.scalar_type());
    TORCH_CHECK(u.scalar_type() == torch::kFloat || u.scalar_type() == torch::kHalf,
                "Unsupported data type: ", u.scalar_type());
    TORCH_CHECK(k.scalar_type() == torch::kFloat || k.scalar_type() == torch::kHalf,
                "Unsupported data type: ", k.scalar_type());
    TORCH_CHECK(v.scalar_type() == torch::kFloat || v.scalar_type() == torch::kHalf,
                "Unsupported data type: ", v.scalar_type());
    TORCH_CHECK(y.scalar_type() == torch::kFloat || y.scalar_type() == torch::kHalf,
                "Unsupported data type: ", y.scalar_type());

    return dispatchForwardKernel(B, T, C, w, u, k, v, y);
}

torch::Tensor mps_backward_kernel(int64_t T_max, int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
                                  torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    // Check whether the input tensors reside on the MPS device and whether they are contiguous.
    TORCH_CHECK(w.device().is_mps() && u.device().is_mps() && k.device().is_mps() && v.device().is_mps() && y.device().is_mps()
                && gy.device().is_mps() && gw.device().is_mps() && gu.device().is_mps() && gk.device().is_mps() && gv.device().is_mps(),
                "All tensors must be MPS tensors");
    TORCH_CHECK(w.is_contiguous() && u.is_contiguous() && k.is_contiguous() && v.is_contiguous() && y.is_contiguous()
                && gy.is_contiguous() && gw.is_contiguous() && gu.is_contiguous() && gk.is_contiguous() && gv.is_contiguous(),
                "All tensors must be contiguous");

    // Check the supported data types for the backward kernel.
    TORCH_CHECK(w.scalar_type() == torch::kFloat || w.scalar_type() == torch::kHalf,
                "Unsupported data type: ", w.scalar_type());
    TORCH_CHECK(u.scalar_type() == torch::kFloat || u.scalar_type() == torch::kHalf,
                "Unsupported data type: ", u.scalar_type());
    TORCH_CHECK(k.scalar_type() == torch::kFloat || k.scalar_type() == torch::kHalf,
                "Unsupported data type: ", k.scalar_type());
    TORCH_CHECK(v.scalar_type() == torch::kFloat || v.scalar_type() == torch::kHalf,
                "Unsupported data type: ", v.scalar_type());
    TORCH_CHECK(y.scalar_type() == torch::kFloat || y.scalar_type() == torch::kHalf,
                "Unsupported data type: ", y.scalar_type());
    TORCH_CHECK(gy.scalar_type() == torch::kFloat || gy.scalar_type() == torch::kHalf,
                "Unsupported data type: ", gy.scalar_type());
    TORCH_CHECK(gw.scalar_type() == torch::kFloat || gw.scalar_type() == torch::kHalf,
                "Unsupported data type: ", gw.scalar_type());
    TORCH_CHECK(gu.scalar_type() == torch::kFloat || gu.scalar_type() == torch::kHalf,
                "Unsupported data type: ", gu.scalar_type());
    TORCH_CHECK(gk.scalar_type() == torch::kFloat || gk.scalar_type() == torch::kHalf,
                "Unsupported data type: ", gk.scalar_type());
    TORCH_CHECK(gv.scalar_type() == torch::kFloat || gv.scalar_type() == torch::kHalf,
                "Unsupported data type: ", gv.scalar_type());

    return dispatchBackwardKernel(T_max, B, T, C, w, u, k, v, y, gy, gw, gu, gk, gv);
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mps_forward_kernel", &mps_forward_kernel, "MPS forward kernel");
    m.def("mps_backward_kernel", &mps_backward_kernel, "MPS backward kernel");
}