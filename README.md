# Journey to SLAM

This Repository create a [**site**](https://lsxiang.github.io/Journey2SLAM "https://lsxiang.github.io/Journey2SLAM") to note Knowledge point during learning [**SLAM**](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping "Simultaneous Localization and Mapping"). 

The following content will be included in the [**site**](https://lsxiang.github.io/Journey2SLAM "https://lsxiang.github.io/Journey2SLAM") :  
- Introduction to Computer Vision
- Multiple View Geometry
- State Estimation for Robotics
- Popular project introduction
- Document recommendation
- ......

## Participate in editing

The Markdown source for all articles in the  [**site**](https://lsxiang.github.io/Journey2SLAM "https://lsxiang.github.io/Journey2SLAM") is open sourced in the [**Repo/docs**](https://github.com/LSXiang/Journey2SLAM/tree/master/docs) folder, and all pages of the  [**site**](https://lsxiang.github.io/Journey2SLAM "https://lsxiang.github.io/Journey2SLAM") are rendered based on these Markdown files via [**MkDocs**](https://www.mkdocs.org/ "https://www.mkdocs.org/") to generate HTML files for direct access.

Personal ability is limited, and like-minded friends are welcome to write together. You can edit blogs, notes, etc. written during learning [**SLAM**](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping "Simultaneous Localization and Mapping") through Markdown and push them to the [docs](https://github.com/LSXiang/Journey2SLAM/tree/master/docs) folder in this Repository. You can also modify the appropriate file and submit a `Pull Request` , or just issue [***Issues***](https://github.com/LSXiang/Journey2SLAM/issues) for your specific situation.

The following is a local simulation of the **Ubuntu** system to run this website, other systems can install the corresponding plug-in and run according to the following environment requirements.

#### 1. Prerequisites

You need install `python3` , `git`, [`Python-Markdown`](https://python-markdown.github.io/),  [`MkDocs`](https://www.mkdocs.org/),  [`PyMdown Extensions`](https://facelessuser.github.io/pymdown-extensions), [`Pygments`](http://pygments.org/). Open one terminal and switch to the path where you want to download the project :

```
sudo apt-get install python3 git python3-pip
git clone https://github.com/LSXiang/Journey2SLAM.git
cd Journey2SLAM/
python3 -m pip install -r requirements.txt
```

#### 2. Build

```
python3 -m mkdocs serve
```

Then visit http://127.0.0.1:8000 in your local browser to see the website performance.

**NOTE: Please be sure to sign the article. If you are authorizing to reprint the articles on your personal website, please also include the original source in the article. When you initiate a submission, you will agree to the "[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.zh)" knowledge used by this site. Share the agreement, please read the terms of the agreement before submitting the manuscript to determine whether you accept the agreement. At the same time, you can apply to us to revoke the authorization to publish the article at any time. You only need to apply for the Pull Request of the corresponding file in Repo.** 
[![license](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)



---

# SLAM之旅

这个 [**Repo**](https://github.com/LSXiang/Journey2SLAM "https://github.com/LSXiang/Journey2SLAM") 创建一个[**网站**](https://lsxiang.github.io/Journey2SLAM "https://lsxiang.github.io/Journey2SLAM")用于记录学习 [**SLAM**](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping "Simultaneous Localization and Mapping") 过程中的知识点。

该[**网站**](https://lsxiang.github.io/Journey2SLAM "https://lsxiang.github.io/Journey2SLAM")将包括一下内容：  

- 计算机视觉基础
- 多视几何
- 状态估计
- 开源项目梳理
- 文档推荐
- ……

## 共同编辑

该[**网站**](https://lsxiang.github.io/Journey2SLAM "https://lsxiang.github.io/Journey2SLAM")中的全部文章的 Markdown 源码开源于 [**Repo/docs**](https://github.com/LSXiang/Journey2SLAM/tree/master/docs) 文件夹中，而[**网站**](https://lsxiang.github.io/Journey2SLAM "https://lsxiang.github.io/Journey2SLAM")的所有页面均基于这些 Markdown 文件通过 [**MkDocs**](https://www.mkdocs.org/ "https://www.mkdocs.org/") 进行渲染生成 HTML 文件可直接访问。

个人能力有限，欢迎志同道合的朋友一起写作。 您可以将学习 SLAM 期间编写的博客、笔记等通过 Markdown 编辑并推送到此项目的 [**docs**](https://github.com/LSXiang/Journey2SLAM/tree/master/docs) 文件中。您也可以修改相应的文件然后提交 `Pull Request`，或者仅针对具体情况提出 [***Issues***](https://github.com/LSXiang/Journey2SLAM/issues) 。

下面给出 **Ubuntu** 系统下本地模拟运行本网站，其他系统可以根据下文环境需求安装对应插件并运行。

#### 1. 需求

你需要安装 `python3` , `git`, [`Python-Markdown`](https://python-markdown.github.io/),  [`MkDocs`](https://www.mkdocs.org/),  [`PyMdown Extensions`](https://facelessuser.github.io/pymdown-extensions), [`Pygments`](http://pygments.org/)。打开一个终端，并切换到您希望下载该项目的路径下：

```
sudo apt-get install python3 git python3-pip
git clone https://github.com/LSXiang/Journey2SLAM.git
cd Journey2SLAM/
python3 -m pip install -r requirements.txt
```

#### 2. 运行

```
python3 -m mkdocs serve
```

然后在本地浏览器访问 http://127.0.0.1:8000 ，查看网站效果。



**注意：请务必在文章中署名，若您是授权将您个人网站中的文章转载在本站，也请一并在文中附上原文出处。当您发起投稿后，将意味着同意本站所使用的 "[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.zh)" 知识共享协议，投稿前请先阅读协议条款，确定您是否接受这一协议。同时，您随时可以向我们申请撤销刊登文章的授权，只需要在 Repo 中申请删除对应文件的 Pull Request 即可。**
[![license](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.zh)

