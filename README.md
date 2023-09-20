# sd webui sampler Heun++
## Introduce
Use higher order Runge-Kutta method in the sampler 

- more time for higher quality
- decreasing weight to avoid noises like heun

10 - 20 steps are enough in most cases; almost no differences with euler when steps more than 80

## Samplers
- Heun++test0 is just more prediction steps than Heun 
- Heun++test1 and Heun++test2(recommended) further explore the balance between heun and euler

## Example
![](https://github.com/Carzit/sd-webui-sampler-heunpp/blob/main/images/example.png)
