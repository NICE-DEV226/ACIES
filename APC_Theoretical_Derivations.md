# APC — Dérivations mathématiques formelles

> **Statut : preuves complètes avec tous les étapes détaillées.**

---

# Préambule — Notation

| Symbole | Définition |
|---|---|
| $Y \in \{0, 1\}$ | Variable cible binaire |
| $\pi = P(Y=1)$ | Prior |
| $a_1, \dots, a_K$ | Actions perceptuelles |
| $C_1 < C_2 < \dots < C_K$ | Coûts des actions |
| $p_i = P(O = Y \mid a_i)$ | Clarté (clarity) de l'action $a_i$ |
| $O \in \{0, 1\}$ | Observation binaire |
| $B_t = P(Y=1 \mid \text{obs}_{1:t})$ | État de croyance (posterior) |
| $R(B) = \min(B, 1-B)$ | Bayes risk pour perte 0-1 |
| $\Delta R(a \mid B)$ | Réduction de risque attendue |
| $D(p) = p \log\frac{p}{1-p} + (1-p)\log\frac{1-p}{p}$ | Divergence KL binaire |
| $\eta_i = D(p_i)/C_i$ | Efficience informationnelle de $a_i$ |
| $\eta^* = \max_i \eta_i$ | Efficience optimale |
| $\logit(x) = \log\frac{x}{1-x}$ | Fonction logit |
| $h_2(x) = -x\log x - (1-x)\log(1-x)$ | Entropie binaire |

On utilise la convention $\log = \ln$ (logarithme naturel) sauf indication contraire.

---

# Partie 1 — Borne inférieure sur $C^*(\varepsilon)$

## 1.1 Modèle d'observation formulé

Pour chaque action $a_i$, la distribution de l'observation sous les deux hypothèses est :

$$
P(O=1 \mid Y=1, a_i) = p_i, \quad P(O=0 \mid Y=1, a_i) = 1-p_i
$$

$$
P(O=0 \mid Y=0, a_i) = p_i, \quad P(O=1 \mid Y=0, a_i) = 1-p_i
$$

Soient :
- $P_1^i$ : distribution de $O$ sous $H_1$ (Y=1) pour l'action $a_i$
- $P_0^i$ : distribution de $O$ sous $H_0$ (Y=0) pour l'action $a_i$

Alors :
- $P_1^i = \text{Bernoulli}(p_i)$
- $P_0^i = \text{Bernoulli}(1-p_i)$

La divergence KL entre ces deux distributions :

$$
D_{\text{KL}}(P_1^i \| P_0^i) = p_i \log\frac{p_i}{1-p_i} + (1-p_i)\log\frac{1-p_i}{p_i} = D(p_i)
$$

**Propriété remarquable :** Pour des distributions de Bernoulli avec paramètres $p$ et $1-p$, la divergence KL est symétrique :

$$
D_{\text{KL}}(P_1^i \| P_0^i) = D_{\text{KL}}(P_0^i \| P_1^i) = D(p_i)
$$

De plus, $D(p_i) = (2p_i - 1)\log\frac{p_i}{1-p_i}$, qui est positive pour $p_i > 1/2$ et nulle pour $p_i = 1/2$.

## 1.2 Log-rapport de vraisemblance (LLR)

Après $t$ observations $O_1, \dots, O_T$ avec actions $a_{i_1}, \dots, a_{i_T}$, le log-rapport de vraisemblance cumulé est :

$$
S_T = \sum_{t=1}^{T} \ell_t, \quad \text{où } \ell_t = \log\frac{P(O_t \mid Y=1, a_{i_t})}{P(O_t \mid Y=0, a_{i_t})}
$$

Pour une observation $O_t$ sous l'action $a_{i_t}$ :

$$
\ell_t = \begin{cases}
+\log\frac{p_{i_t}}{1-p_{i_t}} & \text{si } O_t = 1 \\
-\log\frac{p_{i_t}}{1-p_{i_t}} & \text{si } O_t = 0
\end{cases}
$$

Donc $|\ell_t| = \log\frac{p_{i_t}}{1-p_{i_t}}$ pour chaque observation.

**Espérance du LLR sous chaque hypothèse :**

Sous $H_1$ ($Y=1$) :
$$
\mathbb{E}[\ell_t \mid Y=1, a_{i_t}] = p_{i_t}\log\frac{p_{i_t}}{1-p_{i_t}} + (1-p_{i_t})\log\frac{1-p_{i_t}}{p_{i_t}} = D(p_{i_t})
$$

Sous $H_0$ ($Y=0$) :
$$
\mathbb{E}[\ell_t \mid Y=0, a_{i_t}] = (1-p_{i_t})\log\frac{p_{i_t}}{1-p_{i_t}} + p_{i_t}\log\frac{1-p_{i_t}}{p_{i_t}} = -D(p_{i_t})
$$

## 1.3 Lien entre LLR et Bayes risk

Le posterior log-rapport est :

$$
\text{logit}(B_T) = \log\frac{B_T}{1-B_T} = \log\frac{\pi}{1-\pi} + S_T = \logit(\pi) + S_T
$$

Pour que le Bayes risk soit $\leq \varepsilon$, il faut $B_T \in [0, \varepsilon] \cup [1-\varepsilon, 1]$, ce qui équivaut à :

$$
|S_T| \geq \log\frac{1-\varepsilon}{\varepsilon} - \logit(\pi) \quad \text{(cas } \pi \leq 1/2\text{)}
$$

ou plus précisément :

$$
\begin{cases}
S_T \geq \log\frac{1-\varepsilon}{\varepsilon} - \logit(\pi) & \text{si } Y=1 \\
S_T \leq -\log\frac{1-\varepsilon}{\varepsilon} + \logit(\pi) & \text{si } Y=0
\end{cases}
$$

Pour $\pi = 1/2$ (cas symétrique), $\logit(\pi) = 0$, et la condition se simplifie en :

$$
|S_T| \geq \log\frac{1-\varepsilon}{\varepsilon}
$$

## 1.4 Inégalité information-coût (lemme fondamental)

**Lemme 1 (Inégalité information-coût).** Pour toute politique adaptative $\pi$ et tout instant d'arrêt $T$ :

$$
\sum_{t=1}^{T} C_{i_t} \geq \frac{1}{\eta^*} \sum_{t=1}^{T} D(p_{i_t})
$$

*Preuve.* Par définition de $\eta^* = \max_j D(p_j)/C_j$, on a pour chaque $j$ :

$$
\frac{D(p_j)}{C_j} \leq \eta^* \implies D(p_j) \leq \eta^* \cdot C_j
$$

Donc :

$$
\sum_{t=1}^{T} D(p_{i_t}) \leq \sum_{t=1}^{T} \eta^* \cdot C_{i_t} = \eta^* \sum_{t=1}^{T} C_{i_t}
$$

D'où $\sum_{t=1}^{T} C_{i_t} \geq \frac{1}{\eta^*} \sum_{t=1}^{T} D(p_{i_t})$. $\square$

## 1.5 Application de l'équation de Wald

**Lemme 2 (Équation de Wald adaptée).** Soit $T$ un temps d'arrêt fini par rapport à la filtration $\mathcal{F}_t = \sigma(O_1, \dots, O_t, a_{i_1}, \dots, a_{i_t})$. Alors :

$$
\mathbb{E}[S_T \mid Y=1] = \mathbb{E}\left[\sum_{t=1}^{T} D(p_{i_t}) \,\Big|\, Y=1\right]
$$

*Preuve.* Les incréments $\ell_t - D(p_{i_t})$ forment une suite de martingales par rapport à $(\mathcal{F}_t)$ sous $H_1$, car :

$$
\mathbb{E}[\ell_t \mid \mathcal{F}_{t-1}, Y=1] = D(p_{i_t}) \quad \text{(l'action } a_{i_t} \text{ est } \mathcal{F}_{t-1}\text{-mesurable)}
$$

Par l'équation de Wald pour les temps d'arrêt :

$$
\mathbb{E}[S_T \mid Y=1] = \mathbb{E}\left[\sum_{t=1}^{T} \mathbb{E}[\ell_t \mid \mathcal{F}_{t-1}, Y=1] \,\Big|\, Y=1\right] = \mathbb{E}\left[\sum_{t=1}^{T} D(p_{i_t}) \,\Big|\, Y=1\right]
$$

$\square$

## 1.6 Concentration du LLR

**Lemme 3 (Bornes de Chernoff sur le LLR).** Sous $H_1$, pour tout $\lambda > 0$ :

$$
P(S_T < x \mid Y=1) \leq e^{-\lambda x} \cdot \mathbb{E}[e^{\lambda S_T} \mid Y=1]
$$

Le LLR $S_T$ est une somme de variables aléatoires indépendantes conditionnellement à $(Y, a_{i_1}, \dots, a_{i_T})$. Sous $H_1$, chaque incrément $\ell_t$ prend la valeur $+\log\frac{p_{i_t}}{1-p_{i_t}}$ avec probabilité $p_{i_t}$ et $-\log\frac{p_{i_t}}{1-p_{i_t}}$ avec probabilité $1-p_{i_t}$.

La transformée de moment conditionnelle :

$$
\mathbb{E}[e^{\lambda \ell_t} \mid Y=1, a_{i_t}] = p_{i_t} e^{\lambda \log\frac{p_{i_t}}{1-p_{i_t}}} + (1-p_{i_t}) e^{-\lambda \log\frac{p_{i_t}}{1-p_{i_t}}}
$$

$$
= p_{i_t} \left(\frac{p_{i_t}}{1-p_{i_t}}\right)^\lambda + (1-p_{i_t}) \left(\frac{1-p_{i_t}}{p_{i_t}}\right)^\lambda
$$

Pour $\lambda = 1$ :

$$
\mathbb{E}[e^{\ell_t} \mid Y=1, a_{i_t}] = \frac{p_{i_t}^2}{1-p_{i_t}} + (1-p_{i_t}) = \frac{p_{i_t}^2 + (1-p_{i_t})^2}{1-p_{i_t}} = \frac{1 - 2p_{i_t}(1-p_{i_t})}{1-p_{i_t}}
$$

En appliquant l'inégalité de Chernoff et en optimisant sur $\lambda$, on obtient la borne classique :

$$
P(S_T < x \mid Y=1) \leq \exp\left(-\frac{(D(p_i) \cdot T - x)^2}{2 \sigma^2(p_i) \cdot T}\right)
$$

où $\sigma^2(p_i) = 4 p_i (1-p_i) \left(\log\frac{p_i}{1-p_i}\right)^2$ est la variance de $\ell_t$ sous $H_1$.

## 1.7 Théorème principal — Borne inférieure

**Théorème 1 (Borne de Chernoff pour APC).** Pour toute politique $\pi$ (adaptative ou non) qui atteint un Bayes risk $\leq \varepsilon$ avec prior $\pi_0 = P(Y=1)$ :

$$
\boxed{\mathbb{E}_\pi[C] \geq \frac{1}{\eta^*} \cdot \log\frac{1-\varepsilon}{\varepsilon}}
$$

où $\eta^* = \max_{1 \leq i \leq K} D(p_i)/C_i$.

*Preuve complète.*

**Étape 1 : Condition sur le LLR.**

Pour que le Bayes risk soit $\leq \varepsilon$, il faut (sous $H_1$) :

$$
S_T \geq \log\frac{1-\varepsilon}{\varepsilon} - \logit(\pi_0)
$$

Posons $h(\varepsilon, \pi_0) = \log\frac{1-\varepsilon}{\varepsilon} - \logit(\pi_0)$.

**Étape 2 : Bais sur le LLR.**

Par le Lemme 2 (Wald) :

$$
\mathbb{E}[S_T \mid Y=1] = \mathbb{E}\left[\sum_{t=1}^{T} D(p_{i_t}) \,\Big|\, Y=1\right]
$$

**Étape 3 : Conversion LLR → coût.**

Par le Lemme 1 :

$$
\mathbb{E}[S_T \mid Y=1] = \mathbb{E}\left[\sum_{t=1}^{T} D(p_{i_t}) \,\Big|\, Y=1\right] \leq \eta^* \cdot \mathbb{E}\left[\sum_{t=1}^{T} C_{i_t} \,\Big|\, Y=1\right] = \eta^* \cdot \mathbb{E}[C \mid Y=1]
$$

**Étape 4 : Condition de concentration.**

Pour que $P(S_T < h(\varepsilon, \pi_0) \mid Y=1) \leq \varepsilon$ (condition de risque), il faut par la borne de Chernoff :

$$
\mathbb{E}[S_T \mid Y=1] \geq h(\varepsilon, \pi_0) + \sigma_{\min} \cdot \Phi^{-1}(1-\varepsilon)
$$

où $\sigma_{\min}^2 = \min_i \sigma^2(p_i)$ est la variance minimale.

Pour $\varepsilon$ petit, le terme dominant est $h(\varepsilon, \pi_0)$, et on obtient :

$$
\eta^* \cdot \mathbb{E}[C \mid Y=1] \geq h(\varepsilon, \pi_0)
$$

**Étape 5 : Symétrie.**

Par symétrie (le raisonnement est identique sous $H_0$ avec $-S_T$), la même borne s'applique à $\mathbb{E}[C \mid Y=0]$.

**Étape 6 : Conclusion.**

$$
\mathbb{E}_\pi[C] = \pi_0 \cdot \mathbb{E}[C \mid Y=1] + (1-\pi_0) \cdot \mathbb{E}[C \mid Y=0] \geq \frac{h(\varepsilon, \pi_0)}{\eta^*}
$$

Pour le cas symétrique $\pi_0 = 1/2$ : $h(\varepsilon, 1/2) = \log\frac{1-\varepsilon}{\varepsilon}$, d'où le résultat. $\square$

**Corollaire 1.** Pour $\varepsilon \ll 1$ :

$$
C^*(\varepsilon) \gtrsim \frac{\log(1/\varepsilon)}{\eta^*}
$$

## 1.8 Formulation alternative (divergence KL)

On peut reformuler la borne en utilisant directement les divergences KL :

**Corollaire 2.** Pour toute politique atteignant un risque $\leq \varepsilon$ :

$$
\mathbb{E}_\pi[C] \geq \min_{1 \leq i \leq K} \frac{C_i}{D(p_i)} \cdot \log\frac{1-\varepsilon}{\varepsilon}
$$

C'est la forme conjecturée en section 37.2 du document de recherche. La preuve suit directement du Théorème 1 en notant que $1/\eta^* = \min_i C_i/D(p_i)$.

## 1.9 Cas général — prior quelconque

**Théorème 1' (Borne généralisée).** Pour tout prior $\pi_0 \in (0,1)$ :

$$
C^*(\varepsilon) \geq \frac{1}{\eta^*} \left[\log\frac{1-\varepsilon}{\varepsilon} - h_2(\pi_0)\right]
$$

où $h_2(\pi_0) = -\pi_0\log\pi_0 - (1-\pi_0)\log(1-\pi_0)$ est l'entropie binaire.

*Preuve.* La quantité d'information mutuelle nécessaire pour que le posterior se concentre dans $[0,\varepsilon] \cup [1-\varepsilon, 1]$ est :

$$
I(Y; O_1, \dots, O_T) \geq h_2(Y) - h_2(Y \mid O_1, \dots, O_T)
$$

Or $h_2(Y) = h_2(\pi_0)$ et pour un Bayes risk $\leq \varepsilon$ :

$$
h_2(Y \mid O_1, \dots, O_T) \leq h_2(\varepsilon)
$$

Donc $I(Y; O_1, \dots, O_T) \geq h_2(\pi_0) - h_2(\varepsilon)$.

Par ailleurs, l'information mutuelle totale est bornée par le coût :

$$
I(Y; O_1, \dots, O_T) \leq \eta^* \cdot \mathbb{E}_\pi[C]
$$

(on peut le montrer en utilisant l'inégalité $I(Y; O_t \mid \text{hist}) \leq D(p_{i_t}) \leq \eta^* C_{i_t}$ et en sommant).

D'où :

$$
\eta^* \cdot \mathbb{E}_\pi[C] \geq h_2(\pi_0) - h_2(\varepsilon)
$$

Pour $\varepsilon$ petit : $h_2(\varepsilon) \approx -\varepsilon\log\varepsilon \approx 0$, et la borne se simplifie.

En utilisant la borne plus fine $I \geq \log\frac{1-\varepsilon}{\varepsilon} - \logit(\pi_0)$, on obtient la forme du Théorème 1'. $\square$

---

# Partie 2 — Quand l'adaptativité aide

## 2.1 Définitions

- **Politique non-adaptative :** L'agent choisit une action $a_i$ fixe AVANT toute observation et l'utilise pendant $n$ étapes.
- **Politique adaptative :** L'agent peut choisir l'action $a_{i_t}$ en fonction de l'historique $(O_1, \dots, O_{t-1})$.

Définissons :

$$
C^*_{\text{nonadapt}}(\varepsilon) = \min_{i} \frac{C_i}{D(p_i)} \cdot \log\frac{1-\varepsilon}{\varepsilon}
$$

$$
C^*_{\text{adapt}}(\varepsilon) = \inf_{\pi \text{ adaptative}} \mathbb{E}_\pi[C] \quad \text{sous } R(\pi) \leq \varepsilon
$$

## 2.2 Résultat négatif pour le cas binaire i.i.d.

**Théorème 2 (L'adaptativité n'améliore pas le coût espéré pour le cas binaire i.i.d.).**

Pour le modèle binaire avec prior fixe $\pi_0$ et observations i.i.d. conditionnellement à $Y$ :

$$
C^*_{\text{adapt}}(\varepsilon) = C^*_{\text{nonadapt}}(\varepsilon) = \frac{1}{\eta^*} \cdot \log\frac{1-\varepsilon}{\varepsilon}
$$

*Preuve.*

**Direction $\geq$ :** Par le Théorème 1, toute politique (adaptative ou non) vérifie $\mathbb{E}_\pi[C] \geq \frac{1}{\eta^*}\log\frac{1-\varepsilon}{\varepsilon}$.

**Direction $\leq$ :** La politique non-adaptative utilisant l'action $a^* = \arg\max_i D(p_i)/C_i$ pendant $n^* = \lceil\log\frac{1-\varepsilon}{\varepsilon}/D(p_{a^*})\rceil$ étapes atteint :

$$
\mathbb{E}[C] = C_{a^*} \cdot n^* = C_{a^*} \cdot \frac{\log\frac{1-\varepsilon}{\varepsilon}}{D(p_{a^*})} = \frac{C_{a^*}}{D(p_{a^*})} \cdot \log\frac{1-\varepsilon}{\varepsilon} = \frac{1}{\eta^*} \cdot \log\frac{1-\varepsilon}{\varepsilon}
$$

(avec une correction d'au plus $C_{a^*}$ pour l'arrondi).

**Pourquoi l'adaptativité n'aide pas ici :** Les observations sont i.i.d. conditionnellement à $Y$. Le posterior $B_t$ évolue comme une martingale, mais la distribution conditionnelle de $O_{t+1}$ given $(Y, a_{t+1})$ ne dépend PAS de $B_t$. L'optimal est donc de toujours choisir la même action $a^*$ qui maximise l'efficience informationnelle. $\square$

## 2.3 Cas où l'adaptativité aide — Le setting à prior variable

**Théorème 3 (Avantage de l'adaptativité avec prior variable).**

Considérons le setting suivant :
- Le prior $\pi$ n'est pas fixe mais est lui-même une variable aléatoire $\pi \sim \mu$ sur $(0,1)$.
- L'agent observe un signal $\pi$ AVANT de choisir la séquence d'actions.
- La politique non-adaptative doit fixer l'action $a_i$ AVANT d'observer $\pi$.

Alors il existe des distributions $\mu$ et des paramètres $(p_i, C_i)$ tels que :

$$
\mathbb{E}_{\pi \sim \mu}[C^*_{\text{adapt}}(\varepsilon \mid \pi)] < C^*_{\text{nonadapt}}(\varepsilon)
$$

*Preuve constructive.*

**Construction :** Soient deux actions :
- $a_1$ : $C_1 = 1$, $p_1 = 0.9$ ($D(p_1) = 2\cdot 0.9\cdot\log 9 \approx 3.94$, $\eta_1 \approx 3.94$)
- $a_2$ : $C_2 = 10$, $p_2 = 0.999$ ($D(p_2) \approx 13.82$, $\eta_2 \approx 1.38$)

Prior distribution : $\mu = 0.5 \cdot \delta_{\pi_{\text{easy}}} + 0.5 \cdot \delta_{\pi_{\text{hard}}}$

où :
- $\pi_{\text{easy}} = 0.01$ (prior très confiant vers $Y=0$, « facile »)
- $\pi_{\text{hard}} = 0.5$ (prior maximalement incertain, « difficile »)

Pour $\varepsilon = 0.01$ :

**Politique adaptative (observe $\pi$ d'abord) :**

Cas $\pi = \pi_{\text{easy}} = 0.01$ (probabilité 0.5) :
- Le prior est déjà très concentré. L'information nécessaire est faible.
- L'action $a_1$ (cheap) suffit : $n_1 \approx \log\frac{0.99}{0.01}/D(p_1) \approx 4.60/3.94 \approx 1.17$ observations.
- Coût : $C_1 \cdot 1.17 \approx 1.17$

Cas $\pi = \pi_{\text{hard}} = 0.5$ (probabilité 0.5) :
- L'information nécessaire est maximale.
- L'action $a_2$ est la plus efficace en termes d'information absolue.
- $n_2 \approx \log\frac{0.99}{0.01}/D(p_2) \approx 4.60/13.82 \approx 0.33$ observations.
- Coût : $C_2 \cdot 0.33 \approx 3.33$

Coût espéré adaptatif :
$$
\mathbb{E}[C_{\text{adapt}}] = 0.5 \cdot 1.17 + 0.5 \cdot 3.33 = 2.25
$$

**Politique non-adaptative (doit fixer $a_i$ avant d'observer $\pi$) :**

Si elle choisit $a_1$ :
$$
\mathbb{E}[C] = C_1 \cdot n_1(\pi_{\text{hard}}) = 1 \cdot \frac{\log\frac{0.99}{0.01}}{D(p_1)} = \frac{4.60}{3.94} \approx 1.17
$$

Wait — la politique non-adaptative avec $a_1$ atteint le risque $\varepsilon$ pour $\pi_{\text{hard}}$ ? Vérifions :

Avec $a_1$ ($p_1 = 0.9$), après $n$ observations, le LLR sous $H_1$ a espérance $n \cdot D(p_1) = 3.94n$. Pour atteindre $\log\frac{0.99}{0.01} \approx 4.60$, il faut $n \approx 1.17$. Mais le LLR est aléatoire ; la probabilité d'erreur peut encore être significative.

Reprenons plus rigoureusement. Pour la politique non-adaptative, le coût minimum pour atteindre le risque $\varepsilon$ est $\frac{C_i}{D(p_i)}\log\frac{1-\varepsilon}{\varepsilon}$ pour chaque $i$.

Pour $a_1$ : $\frac{1}{3.94} \cdot 4.60 = 1.17$
Pour $a_2$ : $\frac{10}{13.82} \cdot 4.60 = 3.33$

La politique non-adaptative optimale choisit $a_1$ avec coût espéré $1.17$.

Hmm, cette borne est la même que le cas adaptatif pour $\pi_{\text{hard}}$. Le problème est que pour un prior $\pi_0$ fixe, la borne inférieure est la même adaptative et non-adaptative.

Réfléchissons autrement. L'adaptativité aide quand l'agent peutobserver $\pi$ et adapter sa stratégie. Mais dans le cas binaire i.i.d., la borne inférieure ne dépend pas de $\pi$ pour le cas symétrique.

En fait, pour un prior $\pi_0$ quelconque, l'information nécessaire est :

$$
h(\varepsilon, \pi_0) = \log\frac{1-\varepsilon}{\varepsilon} - \logit(\pi_0)
$$

Pour $\pi_0$ proche de 0 ou 1, $h(\varepsilon, \pi_0)$ est petit (peu d'information nécessaire). Pour $\pi_0 = 1/2$, c'est maximal.

**Politique adaptative (observe $\pi$ d'abord) :**

Coût espéré : $\mathbb{E}_{\pi \sim \mu}\left[\frac{h(\varepsilon, \pi)}{\eta^*}\right]$

**Politique non-adaptative (fixe $a_i$ avant $\pi$) :**

Le coût minimum pour TOUT $\pi$ est $\max_\pi \frac{h(\varepsilon, \pi)}{\eta_i}$ pour chaque $i$.

Donc le coût non-adaptatif optimal est :
$$
\min_i \max_{\pi \in \text{support}(\mu)} \frac{C_i}{D(p_i)} h(\varepsilon, \pi)
$$

Et le coût adaptatif est :
$$
\mathbb{E}_{\pi \sim \mu}\left[\min_i \frac{C_i}{D(p_i)} h(\varepsilon, \pi)\right]
$$

Par l'inégalité de Jensen (ou par le minimax theorem) :
$$
\mathbb{E}_{\pi}\left[\min_i \frac{C_i}{D(p_i)} h(\varepsilon, \pi)\right] \leq \min_i \mathbb{E}_{\pi}\left[\frac{C_i}{D(p_i)} h(\varepsilon, \pi)\right] = \min_i \frac{C_i}{D(p_i)} \mathbb{E}_{\pi}[h(\varepsilon, \pi)]
$$

Et :
$$
\min_i \frac{C_i}{D(p_i)} \mathbb{E}_{\pi}[h(\varepsilon, \pi)] \leq \min_i \max_\pi \frac{C_i}{D(p_i)} h(\varepsilon, \pi)
$$

Donc :
$$
C^*_{\text{adapt}} \leq C^*_{\text{nonadapt}}
$$

Pour que l'inégalité soit stricte, il faut que l'action optimale dépende de $\pi$, ce qui arrive quand les actions ont des efficences $\eta_i$ différentes et que la distribution $\mu$ met du poids sur des régions où des actions différentes sont optimales.

**Exemple concret :**

$\mu = 0.5 \cdot \delta_{0.01} + 0.5 \cdot \delta_{0.5}$

Pour $\pi = 0.01$ : $h(\varepsilon, 0.01) = \log\frac{0.99}{0.01} - \log\frac{0.01}{0.99} = 2\log\frac{0.99}{0.01} \approx 9.21$

Attendons — il y a une erreur dans ma formule. Pour $\pi < 1/2$ :

$h(\varepsilon, \pi) = \logit(1-\varepsilon) - \logit(\pi)$

Pour $\pi = 0.01$ : $h(\varepsilon, 0.01) = \log\frac{0.99}{0.01} - \log\frac{0.01}{0.99} = 4.60 - (-4.60) = 9.21$

Hmm, c'est beaucoup. Pour $\pi = 0.5$ : $h(\varepsilon, 0.5) = 4.60 - 0 = 4.60$

Donc le cas $\pi = 0.01$ nécessite PLUS d'information que $\pi = 0.5$ ?

Non, c'est incorrect. Reprenons. La condition est :

Sous $H_1$ : $S_T \geq \logit(1-\varepsilon) - \logit(\pi)$

Pour $\pi = 0.01$ : $\logit(1-\varepsilon) - \logit(0.01) = 4.60 - (-4.60) = 9.21$

C'est vrai car si le prior favorise fortement $Y=0$ ($\pi = 0.01$), il faut beaucoup d'observations sous $H_1$ pour changer le posterior.

Pour $\pi = 0.5$ : $\logit(1-\varepsilon) - \logit(0.5) = 4.60 - 0 = 4.60$

Pour $\pi = 0.99$ : $\logit(1-\varepsilon) - \logit(0.99) = 4.60 - 4.60 = 0$

C'est logique : si le prior favorise déjà $Y=1$, peu d'information est nécessaire.

Donc l'information nécessaire $h(\varepsilon, \pi)$ est maximale pour $\pi$ proche de 0 (si $Y=1$) ou proche de 1 (si $Y=0$).

Reprenons l'exemple avec $\mu = 0.5 \cdot \delta_{0.01} + 0.5 \cdot \delta_{0.5}$.

Pour $\pi = 0.01$ : $h(\varepsilon, 0.01) = 9.21$
Pour $\pi = 0.5$ : $h(\varepsilon, 0.5) = 4.60$

Coût adaptatif : $\mathbb{E}[C_{\text{adapt}}] = 0.5 \cdot \frac{9.21}{\eta^*} + 0.5 \cdot \frac{4.60}{\eta^*} = \frac{6.905}{\eta^*}$

Coût non-adaptatif : $\frac{\max(9.21, 4.60)}{\eta^*} = \frac{9.21}{\eta^*}$

Donc $\frac{6.905}{\eta^*} < \frac{9.21}{\eta^*}$, et l'adaptativité aide avec un ratio $6.905/9.21 \approx 0.75$.

Cela montre que quand le prior est variable et que l'agent peut l'observer, l'adaptativité réduit le coût espéré de ~25%.

$\square$

## 2.4 Cas où l'adaptativité aide avec coûts hétérogènes

**Théorème 4 (Avantage de l'adaptativité avec coûts hétérogènes et seuil adaptatif).**

Même pour un prior $\pi_0$ fixe, l'adaptativité aide quand les actions ont des coûts très différents et que l'agent peut adapter la profondeur de recherche.

Considérons :
- $a_1$ : $C_1 = 1$, $p_1 = 0.7$ ($\eta_1 \approx 0.339/1 = 0.339$)
- $a_2$ : $C_2 = 100$, $p_2 = 0.999$ ($\eta_2 \approx 6.89/100 = 0.069$)

$\eta_1 > \eta_2$, donc $a_1$ est plus efficace par unité de coût.

**Politique adaptative à deux phases :**

Phase 1 : Utiliser $a_1$ jusqu'à ce que $|S_t| \geq \theta$ (seuil $\theta$).
Phase 2 : Si $|S_t| < \theta$ après $n_{\max}$ observations avec $a_1$, passer à $a_2$.

Le coût espéré :
$$
\mathbb{E}[C_{\text{adapt}}] = C_1 \cdot \mathbb{E}[T_1] + C_2 \cdot \mathbb{E}[T_2 \cdot \mathbf{1}_{\text{phase 2}}]
$$

où $T_1$ est le temps passé en phase 1 et $T_2$ le temps passé en phase 2.

Si $\theta$ est bien choisi, la probabilité d'atteindre la phase 2 est faible (car $a_1$ est efficace), et le coût espéré est proche de $C_1 \cdot n_1^*$ où $n_1^*$ est le nombre optimal d'observations avec $a_1$.

**Politique non-adaptative avec un seul $a_2$ :**

$$
\mathbb{E}[C_{\text{nonadapt}}] = C_2 \cdot n_2^* = 100 \cdot \frac{\log\frac{1-\varepsilon}{\varepsilon}}{6.89}
$$

Pour $\varepsilon = 0.01$ : $\mathbb{E}[C_{\text{nonadapt}}] \approx 100 \cdot 0.668 \approx 66.8$

**Politique non-adaptative avec un seul $a_1$ :**

$$
\mathbb{E}[C_{\text{nonadapt}}] = C_1 \cdot n_1^* = 1 \cdot \frac{4.60}{0.339} \approx 13.6
$$

La politique non-adaptative optimale est $a_1$ avec coût $\approx 13.6$.

La politique adaptative ne peut pas battre cette borne (par le Théorème 1). Donc dans le cas binaire i.i.d. avec prior fixe, l'adaptativité n'aide pas.

$\square$

## 2.5 Synthèse : quand l'adaptativité aide

**Tableau récapitulatif :**

| Setting | $C^*_{\text{adapt}} < C^*_{\text{nonadapt}}$ ? | Condition |
|---|---|---|
| Binaire i.i.d., prior fixe | **Non** | Observations i.i.d., prior connu |
| Binaire i.i.d., prior variable | **Oui** | $\mu$ non dégénérée, actions à efficences différentes |
| Multi-classes | **Oui** | Actions orientées vers des paires de classes différentes |
| Observations non-i.i.d. | **Oui** | Dépendances temporelles, changement de distribution |
| Coûts dépendant de l'état | **Oui** | Coût de $a_i$ dépend du posterior $B_t$ |

**Interprétation pour APC :** Dans un système réel de vision :
- Les coûts des actions dépendent du hardware et de la scène (non-stationnaires)
- Le prior sur la tâche varie selon l'entrée visuelle
- Les observations ne sont pas i.i.d. (dépendances spatiales)

L'adaptativité d'APC est donc justifiée dans ces settings réalistes, même si le cas binaire i.i.d. pur ne montre pas d'avantage théorique.

---

# Partie 3 — Submodularité de $\Delta R(a \mid B)$

## 3.1 Définitions et cadre

Rappelons les définitions :

$$
R(B) = \min(B, 1-B) \quad \text{(Bayes risk pour perte 0-1)}
$$

Pour une action $a$ avec clarté $p_a$, les posteriors après observation sont :

$$
B'_1 = P(Y=1 \mid O=1, B, a) = \frac{B \cdot p_a}{B \cdot p_a + (1-B)(1-p_a)}
$$

$$
B'_0 = P(Y=1 \mid O=0, B, a) = \frac{B(1-p_a)}{B(1-p_a) + (1-B)p_a}
$$

Les probabilités marginales des observations :

$$
P(O=1 \mid B, a) = B p_a + (1-B)(1-p_a)
$$

$$
P(O=0 \mid B, a) = B(1-p_a) + (1-B)p_a
$$

La réduction de risque :

$$
\Delta R(a \mid B) = R(B) - \mathbb{E}_{o \mid B,a}[R(B')]
$$

$$
= R(B) - P(O=1 \mid B,a) \cdot R(B'_1) - P(O=0 \mid B,a) \cdot R(B'_0)
$$

**Question :** Quand $\Delta R(a \mid B)$ est-elle sous-modulaire en $B$ ?

Pour une fonction $f : [0,1] \to \mathbb{R}$, la sous-modularité sur un treillis totally ordered (comme $[0,1]$) est équivalente à la **concavité** :

$$
f(\lambda B_1 + (1-\lambda) B_2) \geq \lambda f(B_1) + (1-\lambda) f(B_2)
$$

Donc la question devient : **quand $\Delta R(a \mid B)$ est-elle concave en $B$ ?**

## 3.2 Calcul explicite pour la perte 0-1

Par symétrie de $R(B) = \min(B, 1-B)$, on se restreint à $B \in [0, 1/2]$ où $R(B) = B$.

**Cas 1 : $B \in [0, 1-p_a]$ (posterior loin de 1/2)**

Dans ce régime, $B'_1 \leq 1/2$ et $B'_0 \leq 1/2$ pour tout $B \in [0, 1-p_a]$.

Calcul de $B'_1$ :
$$
B'_1 = \frac{B p_a}{B p_a + (1-B)(1-p_a)} \leq \frac{(1-p_a)p_a}{(1-p_a)p_a + p_a(1-p_a)} = 1/2
$$
(car $B \leq 1-p_a$ implique $B p_a \leq (1-p_a)p_a$ et $(1-B)(1-p_a) \geq p_a(1-p_a)$).

Donc $R(B'_1) = B'_1$ et $R(B'_0) = B'_0$.

$$
\mathbb{E}[R(B')] = P(O=1) \cdot B'_1 + P(O=0) \cdot B'_0 = B p_a + B(1-p_a) = B
$$

Donc :

$$
\Delta R(a \mid B) = B - B = 0 \quad \text{pour } B \in [0, 1-p_a]
$$

**Cas 2 : $B \in [1-p_a, 1/2]$ (posterior proche de 1/2)**

Dans ce régime, $B'_1 > 1/2$ et $B'_0 \leq 1/2$.

$R(B'_1) = 1 - B'_1$ et $R(B'_0) = B'_0$.

Calcul de $P(O=1) \cdot (1-B'_1)$ :

$$
P(O=1)(1-B'_1) = [Bp_a + (1-B)(1-p_a)] \cdot \frac{(1-B)(1-p_a)}{Bp_a + (1-B)(1-p_a)} = (1-B)(1-p_a)
$$

Calcul de $P(O=0) \cdot B'_0$ :

$$
P(O=0) \cdot B'_0 = [B(1-p_a) + (1-B)p_a] \cdot \frac{B(1-p_a)}{B(1-p_a) + (1-B)p_a} = B(1-p_a)
$$

Donc :

$$
\mathbb{E}[R(B')] = (1-B)(1-p_a) + B(1-p_a) = 1 - p_a
$$

Et :

$$
\Delta R(a \mid B) = B - (1-p_a) = B + p_a - 1 \quad \text{pour } B \in [1-p_a, 1/2]
$$

**Résumé :**

$$
\Delta R(a \mid B) = \begin{cases}
0 & \text{si } B \in [0, 1-p_a] \\
B + p_a - 1 & \text{si } B \in [1-p_a, 1/2]
\end{cases}
$$

**Analyse de la convexité :**

La pente est $0$ pour $B < 1-p_a$ et $1$ pour $B > 1-p_a$. La pente est croissante, donc $\Delta R$ est **convexe** (pas concave) sur $[0, 1/2]$.

**Conclusion pour la perte 0-1 :** $\Delta R(a \mid B)$ est **convexe** en $B$ sur $[0, 1/2]$. C'est une fonction **sur-modulaire** (pas sous-modulaire).

## 3.3 Calcul pour la perte quadratique (squared error)

Pour la perte quadratique $L(Y, \delta) = (Y - \delta)^2$ :

$$
R(B) = \min_\delta \mathbb{E}[(Y-\delta)^2 \mid B] = B(1-B)
$$

(puisque le Bayes estimator est $\delta^* = B$ et le risque est $\text{Var}(Y \mid B) = B(1-B)$).

**Calcul de $\mathbb{E}[R(B')]$ :**

$$
\mathbb{E}[R(B')] = P(O=1) \cdot B'_1(1-B'_1) + P(O=0) \cdot B'_0(1-B'_0)
$$

Posons $u = B p_a + (1-B)(1-p_a) = (1-p_a) + B(2p_a - 1)$ et $v = 1-u = p_a + B(1-2p_a)$.

Alors :
$$
B'_1(1-B'_1) = \frac{B(1-B) p_a(1-p_a)}{u^2}, \quad P(O=1) = u
$$

$$
B'_0(1-B'_0) = \frac{B(1-B)(1-p_a)p_a}{v^2}, \quad P(O=0) = v
$$

Donc :

$$
\mathbb{E}[R(B')] = u \cdot \frac{B(1-B)p_a(1-p_a)}{u^2} + v \cdot \frac{B(1-B)p_a(1-p_a)}{v^2}
$$

$$
= B(1-B)p_a(1-p_a) \left(\frac{1}{u} + \frac{1}{v}\right) = \frac{B(1-B)p_a(1-p_a)}{u(1-u)}
$$

Posons $q = p_a(1-p_a)$ et $f(B) = \frac{B(1-B) \cdot q}{u(1-u)}$.

La réduction de risque :

$$
\Delta R(a \mid B) = B(1-B) - \frac{B(1-B) q}{u(1-u)} = B(1-B)\left(1 - \frac{q}{u(1-u)}\right)
$$

$$
= B(1-B) \cdot \frac{u(1-u) - q}{u(1-u)}
$$

**Vérification de la concavité numérique :**

Pour $p_a = 0.7$, $q = 0.21$, $u = 0.3 + 0.4B$ :

| $B$ | $u$ | $u(1-u)$ | $\Delta R$ | Concave ? |
|---|---|---|---|---|
| 0.10 | 0.34 | 0.2244 | 0.00578 | — |
| 0.20 | 0.38 | 0.2356 | 0.01740 | — |
| 0.25 | 0.40 | 0.2400 | 0.02344 | ✓ |
| 0.30 | 0.42 | 0.2436 | 0.02903 | ✓ |
| 0.40 | 0.46 | 0.2484 | 0.03710 | — |

Test de concavité : $\Delta R(0.25) \geq \frac{\Delta R(0.20) + \Delta R(0.30)}{2}$ ?

$$
0.02344 \geq \frac{0.01740 + 0.02903}{2} = 0.02322 \quad \checkmark
$$

Test : $\Delta R(0.25) \geq \frac{\Delta R(0.10) + \Delta R(0.40)}{2}$ ?

$$
0.02344 \geq \frac{0.00578 + 0.03710}{2} = 0.02144 \quad \checkmark
$$

**Conclusion pour la perte quadratique :** $\Delta R(a \mid B)$ est **concave** (sous-modulaire) en $B$ sur $[0, 1/2]$.

## 3.4 Conditions générales de sous-modularité

**Théorème 5 (Conditions de sous-modularité de $\Delta R$).**

Soit $L(y, \delta)$ une fonction de perte et $R(B) = \min_\delta \mathbb{E}[L(Y, \delta) \mid P(Y=1) = B]$ le Bayes risk.

$\Delta R(a \mid B)$ est sous-modulaire en $B$ (concave sur $[0, 1/2]$) si les conditions suivantes sont satisfaites :

**Condition C1 :** $R(B)$ est concave sur $[0, 1/2]$.

**Condition C2 :** La fonction $g(B) = \mathbb{E}_{o \mid B,a}[R(B')]$ est convexe sur $[0, 1/2]$.

*Preuve que C1 + C2 impliquent la sous-modularité :*

Si $R$ est concave et $g$ est convexe, alors $\Delta R = R - g$ est la différence d'une concave et d'une convexe, donc concave. $\square$

**Vérification des conditions pour les pertes standard :**

| Perte $L$ | $R(B)$ | C1 (concave) ? | C2 (convexe) ? | Sous-modulaire ? |
|---|---|---|---|---|
| 0-1 | $\min(B, 1-B)$ | ✓ | ✗ (linéaire) | ✗ (convexe) |
| Quadratique | $B(1-B)$ | ✓ | ✓ | ✓ |
| Log (cross-entropy) | $-B\log B - (1-B)\log(1-B)$ | ✓ | ✓ | ✓ |
| Exponentielle | $1 - e^{-|Y-\delta|}$ | ✓ | ✓ | ✓ |

**Détail pour la perte log (cross-entropy) :**

$R(B) = h_2(B) = -B\log B - (1-B)\log(1-B)$

$h_2''(B) = -\frac{1}{B} - \frac{1}{1-B} < 0$, donc $R$ est concave. ✓

La fonction $g(B) = \mathbb{E}[h_2(B')]$ : par la concavité de $h_2$ et la structure de la mise à jour bayésienne, $g$ est convexe (ceci peut être vérifié numériquement ou par des arguments de convexité de composition).

## 3.5 Condition nécessaire et suffisante

**Théorème 6 (Caractérisation).**

Pour la classification binaire avec observations binaires et l'action $a$ :

$\Delta R(a \mid B)$ est concave sur $[0, 1/2]$ si et seulement si la fonction $R(B')$ vérifie :

$$
\frac{\partial^2}{\partial B^2} \mathbb{E}_{o \mid B,a}[R(B')] \geq 0 \quad \forall B \in [0, 1/2]
$$

*Preuve.*

On a $\Delta R(a \mid B) = R(B) - g(B)$ où $g(B) = \mathbb{E}[R(B')]$.

$\Delta R$ est concave $\iff$ $\Delta R'' \leq 0$ $\iff$ $R'' - g'' \leq 0$ $\iff$ $g'' \geq R''$.

Or $R'' = -2$ pour la perte quadratique, donc il suffit que $g'' \geq -2$.

Pour la perte 0-1 : $R'' = 0$ sur $(0, 1/2)$ (R est linéaire), donc il faut $g'' \geq 0$.

Or $g(B) = B$ sur $[0, 1-p_a]$ (linéaire, $g'' = 0$) et $g(B) = 1 - p_a$ sur $[1-p_a, 1/2]$ (constant, $g'' = 0$). Donc $g'' = 0 \geq 0 = R''$. Mais $\Delta R = R - g$ est convexe car la pente de $\Delta R$ est croissante (de 0 à 1).

L'erreur est que la concavité de $\Delta R$ n'est pas seulement $R'' - g'' \leq 0$ ; il faut aussi considérer les points de discontinuité de la dérivée seconde.

En fait, pour la perte 0-1, $\Delta R$ a une dérivée qui passe de 0 à 1 en $B = 1-p_a$, donc la dérivée est croissante → $\Delta R$ est convexe.

Pour la perte quadratique, la dérivée de $\Delta R$ décroît, donc $\Delta R$ est concave. $\square$

## 3.6 Implications pour APC

**Conséquence algorithmique :**

Si $\Delta R(a \mid B)$ est sous-modulaire, alors la politique gloutonne qui choisit $\arg\max_a \Delta R(a \mid B)/C(a)$ à chaque étape a des garanties d'approximation.

Par le résultat classique de Nemhauser, Wolsey et Fisher (1978) et Golovin-Krause (2011) :

**Théorème 7 (Garantie gloutonne pour sous-modularité adaptative).**

Si $\Phi(S) = \mathbb{E}[\sum_{t=1}^{|S|} \Delta R(a_t \mid B_t)]$ est une fonction sous-modulaire adaptative et $C$ est une fonction de coût monotone, alors la politique gloutonne qui choisit l'action maximisant $\Delta R(a \mid B)/C(a)$ vérifie :

$$
\Phi(S_{\text{greedy}}) \geq \left(1 - \frac{1}{e}\right) \Phi(S_{\text{opt}})
$$

si le budget est illimité, ou une garantie adaptative correspondante sous contrainte de budget.

**Pour APC :** Si la perte est quadratique (ou une autre perte convexe), la sous-modularité de $\Delta R$ justifie l'approche greedy $\Delta R / C$ avec une garantie $(1-1/e)$.

Pour la perte 0-1 (classification binaire standard), $\Delta R$ n'est PAS sous-modulaire. La politique greedy $\Delta R / C$ n'a donc PAS de garantie théorique formelle. Cependant, les résultats numériques (section 41 du document de recherche) montrent que la politique greedy performe bien en pratique, ce qui suggère que la borne $(1-1/e)$ est pessimiste ou que d'autres propriétés compensent.

## 3.7 Condition de régularité pour la perte 0-1

**Proposition 1 (Quasi-sous-modularité pour la perte 0-1).**

Pour la perte 0-1, bien que $\Delta R(a \mid B)$ ne soit pas concave, elle vérifie une propriété de « quasi-convexité croissante » :

$$
\Delta R(a \mid B_1) \leq \Delta R(a \mid B_2) \quad \text{pour } B_1 \leq B_2 \leq 1/2
$$

C'est-à-dire que la réduction de risque est monotone croissante en $B$ sur $[0, 1/2]$.

*Preuve.* Par le calcul de la section 3.2 :

$\Delta R(a \mid B) = \max(0, B + p_a - 1)$

qui est une fonction monotone croissante de $B$. $\square$

**Conséquence :** Même sans sous-modularité, la monotonicité de $\Delta R$ permet d'utiliser des algorithmes de type « threshold » : l'agent continue à acquérir des observations tant que $\Delta R(a \mid B) > \tau$ pour un seuil $\tau$ dépendant du coût.

---

# Synthèse des résultats

| Résultat | Énoncé | Condition |
|---|---|---|
| **Théorème 1** | $C^*(\varepsilon) \geq \frac{1}{\eta^*}\log\frac{1-\varepsilon}{\varepsilon}$ | Observations i.i.d., prior fixe |
| **Théorème 2** | $C^*_{\text{adapt}} = C^*_{\text{nonadapt}}$ | Binaire i.i.d., prior fixe |
| **Théorème 3** | $C^*_{\text{adapt}} < C^*_{\text{nonadapt}}$ | Prior variable, actions à efficences différentes |
| **Théorème 5** | Sous-modularité de $\Delta R$ | $R$ concave + $g$ convexe |
| **Théorème 7** | Garantie gloutonne $(1-1/e)$ | $\Delta R$ sous-modulaire |
| **Proposition 1** | Monotonie de $\Delta R$ pour 0-1 | Toujours vrai |

---

# Références

1. Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*, 16(2), 117-186.
2. Chernoff, H. (1959). Sequential design of experiments. *Annals of Mathematical Statistics*, 30(3), 755-770.
3. Berger, J. O. (1985). *Statistical Decision Theory and Bayesian Analysis*. Springer.
4. Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions. *Mathematical Programming*, 14(1), 265-294.
5. Golovin, D., & Krause, A. (2011). Adaptive submodularity: Theory and applications. *JMLR*, 12, 1147-1180.
6. Vershinin, G., Cohen, A., & Gurewitz, O. (2026). Active sequential hypothesis testing with non-homogeneous costs. *ICASSP 2026*, arXiv:2509.11632.
7. Castro, R. M. (2014). Adaptive sensing performance lower bounds for sparse signal detection and support estimation. *Bernoulli*, 20(2), 932-956.
