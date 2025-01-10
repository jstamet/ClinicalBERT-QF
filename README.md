# ClinicalBERT-QF
This repo is a set of experiments investigating how query-free attacks can compromise LLMs trained to perform de-identification of sensitive healthcare information. This project illustrates how query-free attacks that have been implemented by Zhuang et al (https://github.com/OPTML-Group/QF-Attack/) on Text-To-Image Models like StableDiffusion can also be used against masked language models like ClinicalBERT.

<div align="center">
  <img src="/images/homoglyph.PNG" alt="Attack Methods on Clinical Text">
  <p><em>Figure 1: Demonstration of homoglyph-based attack methods on clinical text data. Text perturbations are highlighted in red</em></p>
</div>
<div align="center">
  <img src="/images/AverageF1.png" alt="Attack Methods on Results">
  <p><em>Figure 2: Average F1 scores showing the effectiveness of different attack methods</em></p>
</div>

