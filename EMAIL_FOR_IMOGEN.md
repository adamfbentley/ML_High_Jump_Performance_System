# What This Is & What I Need From You
### Adam → Imogen

Hey Imo,

So this is the current state of the project. Basically just done the architecture. 

I've packaged up everthing so u can have a look through and see where we're at. You don't need to run any of it yet — this is more so you can see the structure, understand what each piece does, and give me a sanity check. 

---

## Quick Version

I've built the full pipeline from video to recommendations:

**Video** → **Pose Estimation** (extracts your body position each frame) → **Kinematics** (breaks the jump into phases, calculates metrics like takeoff angle, CoM height, GRF) → **Physics Model** (neural network that learns the relationship between technique and jump height, constrained by real physics) → **Optimiser** (finds what specifc changes to ur technique would add height)

Each of those stages is a working module with real code and tests. The part that's missing is *training data* — I need public biomechanics datasets to train the physics model before we fine-tune it on ur jumps. I've found a few but its been a bit of a nightmare trying to import them.

---

## What's in the Zip

- **`ARCHITECTURE.md`** — detailed walkthrough of every component. Read this first if you want to understand the code structure
- **`src/`** — all the research code
- **`scripts/`** — runnable scripts (downloading data, training, processing video)
- **`tests/`** — 41 automated tests, all passing
- **`experiments/`** — training config files
- **`pyproject.toml`** — project dependencies and metadata

I've left out the `services_scaffold/` folder (web deployment stuff) and the `.venv/` folder (Python environment, massive) to keep the zip small. They're not relevant yet.

---

## Things I'd Love Your Input On

Some of this is athlete-side, some is more scientific — but all of it benefits from u actually understanding both:

### 1. Does the pipeline make sense?
The system breaks a jump into phases: approach → curve → penultimate step → takeoff → flight → landing. Then it measures things like takeoff angle, how fast you're moving vertically at takeoff, ground reaction force, and predicted max height. The phase detection code is in `src/kinematics/phase_detection.py` and the takeoff metrics are in `src/kinematics/takeoff_analysis.py` if you want to look at the actual implementation. **Does that match how you and your coach think about the jump?** Are there phases or metrics im missing that you know matter?

### 2. What technique variables actually matter?
The optimiser (`src/optimization/optimizer.py`) tries to adjust 9 things: approach speed, curve radius, penultimate & last step length, plant angle, takeoff lean, takeoff direction, arm swing timing, and free leg angle. **From your experience, are these the things a coach would actually tweak?** Are any of them not really things you can independently control? Anything obvious I'm missing?

### 3. Movement similarity
I'm training the model on public datasets of people doing various movements (countermovement jumps, drop jumps, sprinting, etc) before we use your actual high jump data. The relevance rankings are in `src/data_pipeline/sample.py` — I've ranked CMJs as the most transferable to high jump after high jump itself (0.9 out of 1.0). **Does that feel right biomechanically?** Like is the force production and joint loading in a CMJ actually similar to what happens in a high jump takeoff, or are they more different than they look?

### 4. The body model
I'm using de Leva (1996) male model for body segment parameters — mass fractions and CoM positions for each segment. The code's in `src/pose_estimation/skeleton/com_estimation.py`. Obviously need to switch to the female model for you, but also — **is de Leva still the right reference?** Are there more recent models or ones that are better suited for athletic populations? And would a generic model be good enough for you or do u think your proportions are different enough that we'd need actual segment measurements?

### 4. Physiological factors
The model currently treats technique as the main thing driving jump height — but obviously theres a biological side too. Muscle fibre composition, rate of force development, fatigue effects across a session, even things like how your body responds differently on different days. **From your biomedical perspective, what physiological variables do you think we should be tracking or at least controlling for?** Even if we cant measure them directly, knowing what confounds to expect would help us design around them.

### 5. Joint angles
I've defined 10 anatomical joint angles in `src/pose_estimation/skeleton/joint_angles.py` — bilateral knee flexion, hip flexion, elbow flexion, shoulder flexion, and ankle dorsiflexion. **Are there other angles that matter for high jump that im not capturing?** Like trunk rotation, hip abduction during flight, anything like that?

### 6. Physiological factors
### 7. Video setup we'd want 2+ camera angles, decent lighting, and the bar visible. **Is that doable at your training venue?** Any constraints I should know about — like can we only film from certain angles, or is the lighting rubbish indoors, that kind of thing.

### 8. Study design & protocol
This is probably where you can contribute the most scientifically. When we collect data from your jumps we need an actual protocol — how many jumps per session, how many sessions, what bar heights, standardised warmup, rest intervals between attempts, time of day controls, etc. **How would you design this?** You understand both the research methodology side and the practical reality of how many max-effort jumps u can do before fatigue becomes a confounding variable. I'd also want your input on what we report — like whether we need inter-session reliability metrics, how we handle failed attempts, what our inclusion/exclusion criteria look like.

### 9. Literature gap
I've done a decent amount of reading on PINNs and GNNs but I haven't gone deep on the sports science and biomechanics litreature. Things like what's already been done with markerless pose estimation in athletics, what the known predictors of high jump performance are, whether anyone has tried ML-based technique optimisation before. **Would u be up for helping with the lit review on that side?** Your access to journals and your ability to critically appraise studies would be really helpful here — I can find the ML papers but I'm not the best judge of whether a sports science study is actually well designed.

---

## What I Need From You (When You Have Time)

1. **Have a read through `ARCHITECTURE.md`** — and if you want to look at the actual code, the main modules are all in `src/`. Its all Python, should be fairly readable
2. **Answer the questions above** — some are quick gut-reaction things and some might need a bit of thought, no rush on any of it
3. **Poke around the code if u want** — especially the kinematics and pose estimation stuff in `src/kinematics/` and `src/pose_estimation/`. If anything looks wrong in the implementation logic, you'd spot it faster than me for the biomechanics bits
4. **Start thinking about the data collection protocol** — even a rough draft of how you'd structure the sessions would be really useful
5. **Start thinking about filming** — we'll need some videos of your jumps eventually. Doesnt need to be fancy yet, just thinking about what's practical
6. **Your measurements** — height, weight, and if you have them from any previous testing, limb lengths / segment lengths. Otherwise we can estimate from video later
7. **If you know any biomechanics people** — neither of us are pure biomechanists so if theres anyone in ur department or who you've come across who might be interested in advising even casually, that'd be massive

The main thing holding us up right now is downloading the public training data (AddBiomechanics — a big dataset of people doing various movments with full force plate data). Once I've got that and pre-trained the model, the next step is processing your actual jump videos.

---

## The Science Bit (For the Paper)

The novelty here is using physics-informed neural networks for personalised technique optimisation. Basically normal ML would need thousands of high jump videos to learn anything. Because we bake Newtonian mechanics into the training process (the network literally can't predict something that violates F = ma), it can learn from way less data and the predictions are guarenteed to be physically realistic. We pre-train on general movement data from public datasets, then fine-tune on your specific jumps.

Theres also a graph neural network that treats the skeleton as a connected chain — so it knows that changing your ankle affects your knee affects your hip, rather than treating each joint like its independant. This is important because a coach would never tell you to change one thing in isolation.

I think theres a solid paper in this regardless of the results — the methodology is novel and if it works even partially thats interesting. And if we can show it gives recommendations that align with what experienced coaches say, thats a strong validation.

In terms of contributions, I see it as — I handle the ML architecture, the physics constraints, and the core implementation. You bring the high jump biomechanics knowledge, the biomedical perspective on physiological factors, experimental design, lit review on the sports science side, and you can also sanity-check the data analysis pipeline since u know Python (and obviously u are the dataset lol). We both write up. I think theres a natural split where I write the methods/model sections and you take the lead on the introduction, background, and discussion — you'll be better at situating this in the existing sports science litreature and interpreting what the results actually mean. Your study design experience is also going to be critical for the methodology section holding up to peer review.

---

Let me know what you think.

Cheers,
Adam
