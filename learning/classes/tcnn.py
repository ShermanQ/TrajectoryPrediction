#F(n) = F(n-1) + [kernel_size(n)-1] * dilation(n): nth dilated causal convolution layer since input layer
# F(n) = F(n-1) + 2 * [kernel_size(n)-1] * dilation(n): nth residual causal block since input layer

# F'(n) = 1 + 2 * [kernel_size(n)-1] * (2^n -1)
# if kernel size is fixed, and the dilation of each residual block increases exponentially by 2