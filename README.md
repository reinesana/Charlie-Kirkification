# Charlie-Kirkification

**A CV productivity tool that plays Charlie Kirk whenever you doomscroll or lose focus.**

![Charlie Kirk](https://github.com/reinesana/Charlie-Kirkification/blob/main/charlie-kirk-project/assets/spam/image-1.jpg)

---

## Introduction

**Charlie-Kirkification** is a CV productivity tool inspired by the **kirkification** trend on **TikTok**. Designed for laptop-based work only.  

Using your webcam, the program tracks your eye and iris movement in real time to detect when you’re looking down at your phone (aka doomscrolling).

**Note**: this tool does not work for activities like writing, reading books, or other offline tasks since it uses iris movement to detect doomscrolling

---

## How it works

1. Your webcam feed is processed in real time using face mesh + iris tracking  
2. The program checks whether your iris movement suggests you’re looking at your phone
3. If you doomscroll for longer than a set threshold:
   -  Charlie Kirk starts playing
   -  Your screen gets spammed with Charlie Kirk 

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/reinesana/Charlie-Kirkification.git

cd Charlie-Kirkification
```

### 2. Install dependencies
Make sure you have **Python 3.9+** installed on your system.

```bash
pip install -r requirements.txt
```

### 3. Run the program
```bash
python main.py
```

---

## Configuration

You can customize how the system behaves by editing the configuration values in `main.py`.

### Timer Customization
```python
timer = 2.0  # seconds you can look at your phone before triggering the program
```

### Iris detection threshold
Adjust the `l_ratio` and `r_ratio` thresholds to control the sensitivity of iris detection.
```python
if (l_ratio > 0.70) and (r_ratio > 0.70):
```

---

## Contributing

Feel free to contribtue to improve this tool

1. Fork the repository  
2. Create a new branch:
   ```bash
   git checkout -b feat/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push your branch and open a pull request

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.


Lock in and have fun 💪



