/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sample_cublasLt_LtHSHgemmStridedBatchSimple.h"
#include "helpers.h"

int main() {

    std::vector<int> batch_sizes = {1, 32, 64, 128, 256, 512, 1024, 2048}; // M
    // int intermediate = 14336; // N
    // int hidden = 4096; // K
    int intermediate = 6144; // N
    int hidden = 2048; // K

    for (int batch_size : batch_sizes) {
        printf("Running LtHSHMatmul with batch size=%d intermediate=%d hidden=%d\n", 
            batch_size, intermediate, hidden);

        TestBench<__half, __half, float> props(CUBLAS_OP_N, CUBLAS_OP_N, 
            batch_size, intermediate, hidden, 
            2.0f, 0.0f, 4 * 1024 * 1024 * 2, 2);

        props.run([&props] {
            LtHSHgemmStridedBatchSimple(props.ltHandle, props.transa, props.transb, props.m, props.n, props.k, &props.alpha,
                                        props.Adev, props.lda, props.m * props.k, props.Bdev, props.ldb, props.k * props.n,
                                        &props.beta, props.Cdev, props.ldc, props.m * props.n, props.N, props.workspace,
                                        props.workspaceSize);
        });
    }

    return 0;
}