# Concept Steer — Training Examples

Copy-paste ready text pairs for Contrastive and SAE training. Each concept has 10 matched pairs: positive texts embody the concept, negative texts describe the same scene without the aesthetic quality.

Paste the **Positive** block into the "positive_texts" input and the **Negative** block into the "negative_texts" input of the Train Lens (Contrastive) or Train Lens (SAE) nodes.

For CLI usage, save pairs as a JSON file (see [JSON Format](#json-format) at the bottom).

---

## Table of Contents

- [Cinematic](#cinematic)
- [Ethereal](#ethereal)
- [Dark Moody](#dark-moody)
- [Vintage Film](#vintage-film)
- [Minimalist](#minimalist)
- [Vibrant Pop](#vibrant-pop)
- [Custom Concept Ideas](#custom-concept-ideas)
- [Few-Shot Directory Layout](#few-shot-directory-layout)
- [JSON Format](#json-format)
- [Quick CLI Commands](#quick-cli-commands)

---

## Cinematic

**Description:** Hollywood film-like dramatic lighting, composition, and color grading

### Positive Texts

```
A sweeping aerial shot of a misty mountain range at golden hour, with dramatic lens flare cutting through the clouds and warm amber light painting the peaks in cinematic glory
A lone figure silhouetted against a massive neon-lit cityscape at night, rain-slicked streets reflecting bokeh lights in a moody, atmospheric noir composition
An epic wide-angle shot of a medieval castle on a cliff edge during a thunderstorm, lightning splitting the sky, shot on anamorphic lenses with dramatic depth of field
A close-up portrait with shallow depth of field, dramatic rim lighting, and cool blue shadows contrasting warm key light, like a Villeneuve film still
A vast desert landscape at magic hour with a caravan stretching to the horizon, dust particles catching golden light, shot with sweeping crane movement
An underwater scene with beams of light piercing through deep blue water, a diver silhouetted against the surface, beautifully color-graded like a BBC nature documentary
A rainy Tokyo alley at night with neon signs reflecting in puddles, a solitary figure with an umbrella, shot with anamorphic lens distortion and cinematic grain
A dramatic overhead shot of a chess board, with pieces casting long shadows from a single directional light source, extreme contrast and depth
A sweeping tracking shot through a lavish ballroom, crystal chandeliers creating bokeh, warm candlelight on faces, motion blur suggesting camera movement
An abandoned industrial hallway with dusty light shafts from broken windows, the texture of decaying concrete beautifully lit with cool ambient light
```

### Negative Texts

```
A mountain range photograph taken during the day showing peaks and clouds in the sky with normal lighting conditions
A person standing in a city at night with lights visible on buildings and streets that have some wet surfaces
A castle on a hillside during cloudy weather with thunder visible in the distance, taken as a standard landscape photo
A portrait of a person with even studio lighting from multiple angles, standard white balance, and a neutral background
A desert scene during the afternoon showing sand dunes and a group of people walking, taken with a standard lens
An underwater photo showing clear blue water with a swimmer visible near the surface, natural underwater colors
A street in Tokyo at night showing various shop signs and people walking, taken with a standard camera and normal settings
A chess board with pieces on a table, lit by overhead room lighting, showing typical indoor shadows
A ballroom interior with chandeliers and people, taken with standard event photography settings and flash
A corridor in an old building with windows, showing daylight coming in, standard exposure and white balance
```

---

## Ethereal

**Description:** Dreamy, otherworldly, soft-focus with luminous quality and gentle light

### Positive Texts

```
A ghostly figure draped in flowing translucent fabric, floating through a misty forest glade where morning light refracts into prismatic rainbows through dewdrops
An otherworldly garden where bioluminescent flowers pulse with soft blue and violet light, their petals seeming to dissolve into wisps of luminous mist
A celestial being with iridescent wings spread wide, hovering above a mirror-still lake that reflects a sky filled with aurora borealis in pastel colors
A dreamy double-exposure of a woman's profile filled with a blooming cherry blossom forest, soft focus creating a halo of pink light around her silhouette
An ancient temple ruins overgrown with glowing moss and floating particles of golden light, as if the very air is alive with magic and memory
A soft-focus underwater scene where a dancer in flowing white fabric twirls among clouds of luminescent jellyfish in deep blue water
A snow-covered landscape at twilight where the sky transitions from deep indigo to pale rose, every snowflake catching light like tiny floating diamonds
A bride walking through a field of lavender at sunset, her veil caught by wind creating a gossamer wave, backlit by warm golden light that makes everything glow
A mystical cave interior where shafts of light illuminate crystal formations that scatter rainbow refractions across the walls like captured starlight
A child reaching toward fireflies in a dusky meadow, the tiny lights creating a magical constellation around outstretched fingers, soft bokeh everywhere
```

### Negative Texts

```
A person standing in a forest clearing during the morning wearing regular clothing, with trees and some fog visible around them
A garden at night with some flowers that have bright colors, standard garden lighting and a path visible between the plant beds
A person with costume wings standing near a lake with clouds reflected in the water, normal outdoor lighting during evening
A portrait photo of a woman's side view combined with an overlay of tree branches, standard photo editing technique
Old stone ruins covered in moss and grass, photographed during the day with sunlight coming through gaps in the walls
A swimmer underwater in a pool with fluorescent lighting, wearing a white swimsuit, with some sea creatures visible nearby
A winter landscape at dusk showing snow-covered ground and a sky that is getting dark, snowflakes falling normally
A woman in a white dress walking through a purple flower field during sunset, gentle breeze moving her clothing
The inside of a cave with natural light coming through an opening, showing rock formations and mineral deposits on walls
A child playing outside in the evening with some bugs with lights flying around, lawn and trees in the background
```

---

## Dark Moody

**Description:** Dark, atmospheric, high contrast with deep shadows and emotional intensity

### Positive Texts

```
A brooding portrait shrouded in near-darkness, only a sliver of cold blue light revealing furrowed brows and intense eyes, deep shadows swallowing the rest
An abandoned asylum corridor stretching into absolute blackness, a single flickering light creating harsh angular shadows on peeling walls and rusted beds
A rain-battered window with rivulets distorting the view of a solitary street lamp in otherwise total darkness, the glass itself weeping
A forest at midnight where gnarled tree trunks twist like tortured figures, a faint blood-red moon barely illuminating the canopy of dead branches
A smoke-filled underground bar where a single spotlight cuts through the haze to illuminate a worn microphone on an empty stage, all else in shadow
A decayed gothic cathedral interior where darkness pools in every crevice, a single votive candle guttering beside a cracked marble saint
Storm clouds roiling over a dark sea, waves crashing against black rocks, the only light a distant lighthouse beam slashing through sheets of rain
A noir-style street scene in near-total darkness, wet cobblestones reflecting a single red neon sign, a fedora'd silhouette disappearing into an alley
A portrait of weathered hands clutching a faded photograph, lit only from below by a dying ember, extreme chiaroscuro rendering flesh and shadow
A desolate winter landscape under a starless sky, bare trees like black veins against gunmetal grey, the ground frozen and cracked like shattered glass
```

### Negative Texts

```
A portrait of a person with normal indoor lighting, showing their face clearly with standard contrast and balanced exposure settings
A hospital corridor with standard fluorescent lighting, clean floors, and medical equipment visible along the walls under even illumination
A window on a rainy day showing a street outside with normal city lights, the rain visible on the glass with typical indoor lighting
A forest scene at night with moonlight visible through the trees, standard night photography showing trunks and branches with some ground detail
A bar or pub interior with typical lighting, showing a stage area with a microphone, tables, and chairs in standard ambient light
A church interior showing architecture with standard photo lighting, pews, columns, and religious artifacts visible with normal exposure
An ocean scene with rough waves and cloudy sky, a lighthouse in the distance, photographed with standard landscape settings during day
A city street at night showing wet pavement, a neon sign, and people walking, standard street photography with normal exposure
Close-up photo of hands holding an old photograph, taken with flash or standard indoor lighting with visible detail throughout the image
A winter landscape during overcast day showing bare trees and frozen ground, standard exposure showing landscape in grey tones
```

---

## Vintage Film

**Description:** Analog film look with warm tones, grain, light leaks, and nostalgic imperfections

### Positive Texts

```
A sun-drenched summer afternoon captured on Kodak Portra 400, warm amber color cast, visible film grain, a child running through a sprinkler with delicious halation on highlights
A faded Polaroid of a roadside diner at dusk, characteristic color shift to warm yellows and cool shadows, soft focus edges with that unmistakable instant film border
A dreamy double-exposure on expired film, showing a woman's face superimposed over a field of wildflowers, light leaks bleeding orange and magenta across the frame
A candid street photograph with the gritty texture of Tri-X pushed to 1600, deep blacks and bright whites, visible grain structure lending atmosphere to the scene
A 1970s-style family photo with oversaturated Ektachrome colors, slightly off white balance, lens flare from shooting into sunlight, and a soft vignette at the edges
A washed-out beach scene shot on expired Fuji Superia, characteristic green-shift in shadows, faded highlights, and that wonderful pastel quality of degraded emulsion
A late-afternoon portrait with golden Kodachrome warmth, the subject bathed in rich amber tones, razor-sharp yet somehow nostalgic, like a rediscovered family treasure
A moody night scene shot on high-speed film with extreme grain, neon lights creating soft halos, the entire image swimming in that magical photographic texture
A garden party captured on medium format Hasselblad, creamy bokeh, waist-level finder perspective, rich but restrained colors with that unmistakable medium format depth
A rainy window view shot on Cinestill 800T, tungsten-balanced film creating cool blue tones outdoors and warm halation around the streetlights, red halos on highlights
```

### Negative Texts

```
A digital photo of a sunny day with a child playing near a sprinkler, taken with a modern camera with clean, noise-free image quality and accurate colors
A standard digital photo of a diner at sunset, taken with a smartphone showing clean edges, accurate white balance, and no film artifacts or borders
Two photos layered together using digital editing, showing a woman and flowers, clean blend with no color artifacts or light effects from film processing
A street photo taken with a digital camera, clean black and white conversion with smooth tones, no visible noise or grain texture in the image
A family photo taken with a digital camera, accurate colors and white balance, no lens flare or color shifts, sharp and evenly exposed throughout
A digital beach photo with accurate sand and water colors, clean exposure, no color shifts or fading, typical modern camera output quality
A portrait taken during golden hour with a digital camera, correct white balance maintaining natural skin tones, clean and sharp throughout the image
A night scene photographed with a modern camera on a tripod, clean long exposure with sharp neon signs, no grain or texture artifacts visible
A garden event photographed with a modern digital medium format camera, clean bokeh and sharp focus, accurate color reproduction with no film effects
A photo through a rainy window taken with digital camera, clean image with correct white balance, no color casts or halation around light sources
```

---

## Minimalist

**Description:** Clean, sparse compositions with lots of negative space, simple forms, and restrained palette

### Positive Texts

```
A single white feather resting on an infinite plane of pale grey, casting a whisper-thin shadow, the vast emptiness around it lending profound significance
A geometric concrete staircase ascending against a pure white sky, the clean lines and sharp angles creating a study in form and negative space
A solitary tree in a snow-covered field, its bare branches forming a delicate ink-drawing silhouette against a uniform overcast sky, nothing else in frame
A perfectly smooth pebble centered on a sweep of fine sand, its oval shadow the only contrast in the frame, zen-like simplicity made tangible
Two parallel lines of footprints in wet sand stretching to a vanishing point where sea meets pale sky, the composition stripped to its barest elements
A single cup of black coffee on a plain white table, shot from directly above, the dark circle a stark punctuation mark in an ocean of white
A long-exposure of a single wave washing over an empty white beach, the water reduced to a smooth gradient from foam to glass, nothing more
A vertical thin reed reflected perfectly in still water, bisecting the frame into symmetrical halves of sky and mirror, pure geometric tranquility
A small red door set into a massive white wall, the scale contrast and color isolation creating an almost abstract composition of proportion
A single light bulb hanging from a long wire in an empty white room, its glow creating the gentlest gradient on otherwise featureless surfaces
```

### Negative Texts

```
A table with multiple feathers, pens, notebooks, and other objects scattered across a wooden surface, with a detailed patterned background visible
A building with multiple staircases, railings, signs, and windows visible against a complex urban backdrop with other structures and sky elements
A grove of multiple trees in a varied landscape showing grass, bushes, fences, paths, and other vegetation with complex textures throughout
Several rocks and pebbles of different sizes on a beach with seaweed, shells, driftwood, and footprints visible in the sand and surrounding area
A busy beach scene with many people, beach umbrellas, towels, coolers, and various equipment, multiple footprints and activities in the frame
A kitchen counter with multiple coffee cups, utensils, a coffee maker, plates, and various items on a patterned counter with shelves behind
A beach scene with multiple waves, surfers, seabirds, beach grass, and various coastal features creating a complex and detailed composition
A pond with multiple reeds, lily pads, fish, and surrounding forest vegetation reflected in the water with ripples and varied textures visible
A building facade with multiple doors, windows, shutters, and architectural details of different colors and styles in a busy street setting
A room full of furniture, pictures, shelves, lamps, and decorative items with complex lighting from multiple sources creating varied shadows
```

---

## Vibrant Pop

**Description:** Highly saturated, bold colors with graphic punch and visual energy

### Positive Texts

```
An explosion of electric magenta, blazing yellow, and deep cobalt blue paint splashing against a pure white background, colors so intense they vibrate
A tropical parrot in flight, its feathers a riot of scarlet, emerald, and sapphire against a perfectly saturated cyan sky, every color pushed to maximum
A Tokyo street at night transformed into a neon dreamscape, hot pink kanji signs reflecting in puddles alongside electric green and ultraviolet blue
A pop-art style portrait with Andy Warhol-level color saturation, skin in hot pink, hair in electric blue, background in screaming yellow, bold outlines
Stacks of colorful macarons forming a rainbow tower, each layer an impossibly saturated pastel, the light making the colors seem to glow from within
A field of tulips where each row is a different pure, saturated color — crimson, golden, violet, orange — stretching to the horizon like a painter's palette
A vintage car in candy apple red parked against a wall painted in vivid turquoise, the complementary colors creating maximum chromatic energy
Hot air balloons filling the frame in every conceivable color, their geometric panels creating a mosaic of pure hues against an impossibly blue sky
A Mexican Day of the Dead altar covered in marigolds so orange they glow, sugar skulls painted in hot pink and lime green, papel picado in every color
A coral reef underwater scene with maximum color: fluorescent anemones, electric blue tangs, yellow butterfish, and magenta coral pulsing with life
```

### Negative Texts

```
Paint being mixed on a palette showing various muted grey, beige, and brown tones on a neutral background in standard lighting conditions
A bird in flight showing typical coloring against an overcast grey sky, standard nature photography with normal saturation and muted tones
A city street at night showing typical yellow-toned street lamps and some commercial signs, standard urban night photography with normal exposure
A standard portrait with normal skin tones against a grey background, typical studio photography with accurate white balance and natural colors
Pastries arranged on a tray in natural bakery lighting, showing typical muted pastry colors in beige, light brown, and cream tones
A field of flowers during an overcast day showing somewhat muted colors, standard landscape photography with clouds diffusing the light
A car parked on a regular street showing standard automotive paint in a common neutral color against a typical urban background setting
Hot air balloons in the distance during a hazy day, their colors appearing muted and desaturated due to atmospheric conditions and distance
A traditional altar with flowers and decorations in natural indoor lighting, showing warm but normal colors without oversaturation
An underwater photograph of a reef at moderate depth where water has filtered out warm colors, showing mainly blue and grey tones throughout
```

---

## Custom Concept Ideas

Here are starter pairs for concepts you might want to train. These are shorter (5 pairs each) — expand to 10+ for better results.

### Cyberpunk

**Positive:**

```
A rain-soaked neon megacity with holographic advertisements towering over crowded streets, chrome and glass reflecting electric pink and cyan light
A cyborg woman with glowing circuitry visible beneath translucent skin, standing in a data-stream of floating kanji characters and wireframe geometry
An underground hacker den lit by dozens of monitors casting blue light on walls covered in cables, circuit boards, and graffiti of digital skulls
A massive corporate arcology piercing smog-filled clouds, its surface alive with crawling LED patterns, flying vehicles streaming between towers
A street-level ramen stall beneath a tangle of power lines and holographic signs, steam rising into air thick with digital particles and neon glow
```

**Negative:**

```
A city at night showing buildings with some lights and signs, people walking on sidewalks, normal urban photography
A person wearing electronic accessories standing in front of a computer screen, standard portrait lighting
A room with multiple computer monitors and cables, typical office or workspace lighting
A tall modern building photographed from below against a cloudy sky, standard architectural photography
A food stall on a street with overhead wiring, normal evening lighting and some steam from cooking
```

### Watercolor

**Positive:**

```
A landscape where colors bleed and bloom like wet watercolor on textured paper, soft edges dissolving mountains into sky with granulating pigments
A portrait rendered in translucent washes of cerulean and raw sienna, the white of the paper glowing through skin tones, edges lost and found
A garden scene with flowers painted in loose, flowing strokes, colors running into each other at wet boundaries, creating beautiful unpredictable bleeds
An architectural study where a stone bridge emerges from fog in graded washes, the water beneath captured in flowing cobalt with salt-texture effects
A still life of fruit where each piece is a study in transparent glazing, light seeming to pass through the paint itself, edges softly feathered
```

**Negative:**

```
A landscape photograph with sharp detail throughout, clear boundaries between land and sky, standard digital camera output
A portrait with precise skin detail, sharp focus on features, standard studio photography with accurate colors
A garden photographed with standard settings, clear detail on each flower, typical digital nature photography
A bridge photographed with standard lens showing clear architectural detail, normal landscape composition
A photo of fruit on a table with standard still life lighting, sharp focus and accurate color reproduction
```

### Glitch Art

**Positive:**

```
A portrait disintegrating into horizontal scan lines and pixel-sorted cascades, RGB channels splitting apart, data corruption creating alien beauty
A landscape where the image appears to load incorrectly, vertical bars of displaced color data, JPEG artifact aesthetics amplified into art
A figure fragmenting into mosaic blocks of wrong-colored pixels, compression artifacts blooming like digital flowers, the error becoming the medium
A cityscape where databending has stretched buildings into impossible streams of color, horizontal tearing creating a new surreal architecture
A photograph glitching into pure data visualization, hex values bleeding through displaced pixel arrays, the boundary between image and code dissolving
```

**Negative:**

```
A portrait photograph with clean, sharp detail, accurate colors and no digital artifacts, standard camera output
A landscape with correct rendering, no compression artifacts, standard photo with accurate representation
A clear photograph of a person, normal color reproduction, no pixelation or digital errors visible
A city photographed with standard settings, buildings appearing normal, accurate perspective and detail
A photograph displayed correctly on screen, clear image with proper data rendering and normal display
```

---

## Few-Shot Directory Layout

For image-based training with the Train Lens (Few-Shot) node or `few-shot` CLI command:

```
training_data/
├── cyberpunk/
│   ├── positive/          # 10-50 images embodying the concept
│   │   ├── cyber_001.png
│   │   ├── cyber_002.jpg
│   │   └── ...
│   └── negative/          # 10-50 images WITHOUT the concept (same subjects)
│       ├── normal_001.png
│       ├── normal_002.jpg
│       └── ...
└── watercolor/
    ├── positive/
    │   └── ...
    └── negative/
        └── ...
```

**Tips for few-shot:**
- Use 10+ images per side for stable results
- Keep subjects similar between positive and negative (same scenes, different style)
- Supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
- Negative directory is optional — without it, the origin is used as contrast

---

## JSON Format

For the `text-pairs` CLI command, save pairs as a JSON array:

```json
[
  {
    "positive": "A sweeping aerial shot of a misty mountain range at golden hour, with dramatic lens flare...",
    "negative": "A mountain range photograph taken during the day showing peaks and clouds..."
  },
  {
    "positive": "A lone figure silhouetted against a massive neon-lit cityscape at night...",
    "negative": "A person standing in a city at night with lights visible..."
  }
]
```

Then run:

```bash
python tools/lens_factory.py text-pairs my_pairs.json --concept my_concept --target zimage
```

---

## Quick CLI Commands

```bash
# Set encoder path first
export QWEN_ENCODER_PATH="/path/to/qwen_3_4b.safetensors"

# ── Contrastive Training (fast, ~2-5 min) ──────────────────────────

# Train from a preset
python tools/lens_factory.py auto cinematic --target zimage

# Train all presets
python tools/lens_factory.py batch-all --target zimage

# Train from custom pairs
python tools/lens_factory.py text-pairs my_pairs.json --concept cyberpunk --target zimage

# ── SAE Training (interpretable, ~10-20 min) ───────────────

# SAE from preset
python tools/lens_factory.py sae cinematic --target zimage --sae-features 30

# SAE all presets (reuse SAE across concepts)
python tools/lens_factory.py batch-all --target zimage --method sae --sae-save ./my_sae.pt

# SAE with pre-trained SAE (much faster for 2nd+ concept)
python tools/lens_factory.py sae ethereal --sae-load ./my_sae.pt

# SAE without contrastive refinement
python tools/lens_factory.py sae dark_moody --no-refine-contrastive

# ── Few-Shot (from images) ─────────────────────────────────

python tools/lens_factory.py few-shot ./positive_imgs/ --negative ./negative_imgs/ --concept dreamy

# ── Inspect ─────────────────────────────────────────────────

python tools/lens_factory.py list-presets
python tools/lens_factory.py list-lenses
```
