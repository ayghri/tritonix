
DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float32
# torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
# torch.backends.cudnn.allow_tf32 = ALLOW_TF32

# Parameter grid for random selection
param_grid = {
    "N": [1, 2, 4, 8],
    "C": [8, 16, 32, 64],
    "H": [32, 64, 128, 256],
    "W": [32, 64, 128, 256],
    "K": [16, 32, 64, 128],
    "R": [3, 5, 7],
    "S": [3, 5, 7],
    "stride": [1, 2, 3],
    "padding": [0, 1, 2, 3],
    "dilation": [1, 2, 3],
}


def run_random_test():
    # Randomly select parameters from the grid
    # N = random.choice(param_grid["N"])
    # C = random.choice(param_grid["C"])
    # H = random.choice(param_grid["H"])
    # W = random.choice(param_grid["W"])
    # K = random.choice(param_grid["K"])
    # R = random.choice(param_grid["R"])
    # S = random.choice(param_grid["S"])
    # stride_val = random.choice(param_grid["stride"])
    # padding_val = random.choice(param_grid["padding"])
    # dilation_val = random.choice(param_grid["dilation"])
    N = 8
    C = 8
    H = 32
    W = 48
    K = 24
    R = 5
    S = 7
    stride_val = 3
    padding_val = 0
    dilation_val = 2

    stride = (stride_val, stride_val)
    padding = (padding_val, padding_val)
    dilation = (dilation_val, dilation_val)

    print("Testing with parameters:")
    print(f"N={N}, C={C}, H={H}, W={W}, K={K}, R={R}, S={S}")
    print(
        f"stride={stride_val}, padding={padding_val}, dilation={dilation_val}"
    )

    # Create tensors
    torch.manual_seed(2)
    weight = torch.randn(K, C, R, S, device=DEVICE, dtype=DTYPE)
    bias = torch.zeros(K, device=DEVICE, dtype=DTYPE)
    input = torch.randn(N, C, H, W, device=DEVICE, dtype=DTYPE) ** 2

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float32,
        enabled=ALLOW_TF32,
    ):
        # Run the Triton conv2d forward
        conv2d_forward(
            input, weight, bias, stride=stride, padding=padding, dilation=dilation
        )
        # Run both implementations
        torch_out = torch.nn.functional.conv2d(
            input, weight, bias, stride=stride, padding=padding, dilation=dilation
        )

        triton_out = conv2d_forward(
            input, weight, bias, stride=stride, padding=padding, dilation=dilation
        )

        # Check if outputs match
        # if torch.allclose(torch_out, triton_out, atol=1e-4):
        if False:
            # print("✓ Test passed!")
            return True
        else:
            print("✗ Test failed!")
            print(f"Max difference: {torch.max(torch.abs(torch_out - triton_out))}")
            print(f"Norm difference: {torch.norm(torch_out - triton_out)}")
            # show coordinates of argmax
            diff = torch.abs((torch_out - triton_out).float()).detach().cpu().numpy()
            print("Max difference array:", np.max(diff))
            p = np.unravel_index(np.argmax(diff), diff.shape)
            # p = np.unravel_index(np.where(diff> 0.05)[0], diff.shape)

            # p = np.array(p)
            # print("argmax coordinates:", p[:,0])
            # p = p[:,0]
            b, c, h, w = p
            inputs = input[
                b,
                :,
                h * stride_val : h * stride_val + R * dilation_val : dilation_val,
                w * stride_val : w * stride_val + S * dilation_val : dilation_val,
            ]
            print(inputs.shape)
            (print("torch out:", torch_out[p]))
            print("triton_out:", triton_out[p])
            print(
                "manual out:",
                (
                    inputs.float()
                    * weight[
                        c,
                        :,
                    ].float()
                ).sum(),
            )
            return False


# Run multiple random tests
print("Running random convolution tests...")
num_tests = 1
passed = 0

for i in range(num_tests):
    print(f"\nTest {i + 1}/{num_tests}:")
    if run_random_test():
        passed += 1

print(f"\nResults: {passed}/{num_tests} tests passed")
if passed == num_tests:
    print("All tests passed! ✓")
else:
    print("Some tests failed! ✗")
