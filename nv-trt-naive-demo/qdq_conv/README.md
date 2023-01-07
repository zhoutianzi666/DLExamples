

- 他妈的，必须对权重也加上QDQ，否则很有可能会跑fp32的kernel，因为他是真的在跑q + dq + fp32的conv

![在这里插入图片描述](https://img-blog.csdnimg.cn/60d9f8ac9e9341e88bc52351a41f2b21.png)


- 然后我给权重也加上QDQ，果然他真的跑int8了！

![在这里插入图片描述](https://img-blog.csdnimg.cn/ce07023fb59d4199870c9cb8b9d4634e.png)
