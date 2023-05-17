https://towardsdatascience.com/witnessing-the-progression-in-semantic-segmentation-deeplab-series-from-v1-to-v3-4f1dd0899e6e

The idea of atrous (hole) convolution is simple but effective:
    - The regular convolution kernel combines values from a few neighboring pixels to
      calculate the new pixel value.

    - The downside of using such kernel in semantic segmentation is that more local
      relationships, instead of global relationships are considered when we extract features.

    - To improve this, DeepLab V1 borrows an idea from a signal processing algorithm
      to introduce a stride inside the kernel window

    - In this atrous version, we put holes in between each pixel we sample so that we can
      sample a wider range of input with the same kernel size. For example, when the
      atrous rate (dilation rate) is 2 and kernel size is 3x3, rather than taking pixels
      from a 3x3 area, we will take 9 pixels from a 5x5 area by skipping those pixels in between.

    - By using atrous convolution, our 3x3 kernel isn’t as expensive as before
      because we can use fewer kernels to cover a bigger area.
      Also, an atrous convolution over a 28x28 features map can bring a similar global signal
      from a regular convolution over a 7x7 features map. Moreover,
      if we increase the atrous rate, we can effectively use the same computation
      of a 3x3 kernel but achieving much larger field-of-view (FOV).
