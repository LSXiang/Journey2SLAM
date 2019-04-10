
# test

## #TODO

Lorem ipsum dolor sit amet: $p(x|y) = \frac{p(y|x)p(x)}{p(y)}​$


$$
\mathbf{V}_1 \times \mathbf{V}_2 = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
\frac{\partial X}{\partial u} & \frac{\partial Y}{\partial u} & 0 \\
\frac{\partial X}{\partial v} & \frac{\partial Y}{\partial v} & 0 \\
\end{vmatrix}
$$



!!! note "前面需要空4个空格"
    这样才是对的

    这是才是对的


!!! warning ""
    这是hi什么

> 这是什么鬼啊
>
> 你好
>
> > 你好啊

## A setext style header

### A hash style header



<!-- [link](http://example.com){: class="foo bar" title="Some title!" } -->

[link](https://www.baidu.com 'baidu') 

```C++ hl_lines="1 3"
int main(int argc, char** argv) {
  if (argc != 2) {
    return -1;
  }

  return 0;
}
```
*Markdown 源文件*

- <strong>Markdown Preview Enhanced: Toggle</strong>  
  <kbd>ctrl-shift-m</kbd>  
  开／关 Markdown 文件预览。      
- <strong> Markdown Preview Enhanced: Open Mathjax Config </strong>  
  打开 `MathJax` 设置文件。  
- **粗体**
  开关  

### 按键

++ctrl+alt+delete++

- [x] tasklist  


The HTML specification
is maintained by the W3C.

*[HTML]: Hyper Text Markup Language
*[W3C]:  World Wide Web Consortium

!!! note "Phasellus posuere in sem ut cursus"
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

==Apple :==
:   Pomaceous fruit of plants of the genus Malus in
    the family Rosaceae.

Orange
:   The fruit of an evergreen tree of the genus Citrus.

???+ note ""
    ```Bash tab="Bash"
    #!/bin/bash
    STR="Hello World!"
    echo $STR
    ```

    ```C tab="C"
    #include 

    int main(void) {
      printf("hello, world\n");
    }
    ```

    ```C++ tab="C++"
    #include <iostream>

    int main() {
      std::cout << "Hello, world!\n";
      return 0;
    }
    ```

    ```C# tab="C#"
    using System;

    class Program {
      static void Main(string[] args) {
        Console.WriteLine("Hello, world!");
      }
    }
    ```

???+ note "Open styled details"

    ??? danger "Nested details!"
        And more content again.


```hl_lines="1 3" linenums="2"
"""Some file."""
import foo.bar
import boo.baz
import foo.bar.baz
```



[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

First Header  | Second Header
------------- | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell

:smile:这是干嘛

!!! example "Flow Chart Example"

    ````tab="Source"
    ```flow
    st=>start: Start:>http://www.google.com[blank]
    e=>end:>http://www.google.com
    op1=>operation: My Operation
    sub1=>subroutine: My Subroutine
    cond=>condition: Yes
    or No?:>http://www.google.com
    io=>inputoutput: catch something...

    st->op1->cond
    cond(yes)->io->e
    cond(no)->sub1(right)->op1
    ```
    ````

    ```flow tab="Output"
    st=>start: Start:>http://www.google.com[blank]
    e=>end:>http://www.google.com
    op1=>operation: My Operation
    sub1=>subroutine: My Subroutine
    cond=>condition: Yes
    or No?:>http://www.google.com
    io=>inputoutput: catch something...

    st->op1->cond
    cond(yes)->io->e
    cond(no)->sub1(right)->op1
    ```

!!! example "Sequence Diagram Example"

    ````tab="Source"
    ```sequence
    Title: Here is a title
    A->B: Normal line
    B-->C: Dashed line
    C->>D: Open arrow
    D-->>A: Dashed open arrow
    ```
    ````

    ```sequence tab="Output"
    Title: Here is a title
    A->B: Normal line
    B-->C: Dashed line
    C->>D: Open arrow
    D-->>A: Dashed open arrow
    ```

[=0% "0%"]
[=5% "5%"]
[=25% "25%"]
[=45% "45%"]
[=65% "65%"]
[=85% "85%"]
[=100% "100%"]

--8<--
mathjax.txt
uml.txt
--8<--