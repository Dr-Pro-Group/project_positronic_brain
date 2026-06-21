# Zones and Brain Regions

The 3D volume is partitioned into functional **zones** using a deterministic **Voronoi** rule: each neuron is assigned to the nearest zone *seed* (`assign_zones` in `positronic_brain/zones.py`). The rule is geometric rather than learned, so the 3D visualizer always shows a consistent, interpretable layout — and it generalizes cleanly to any number of zones or any grid size.

## The Six Default Zones

| Zone           | Color     | Modality        | Typical role in demos                         |
|----------------|-----------|-----------------|-----------------------------------------------|
| Visual         | `#3B82F6` | sensory         | Fast, focal response to visual drive          |
| Auditory       | `#10B981` | sensory         | Fast, focal response to auditory drive        |
| Somatosensory  | `#F472B6` | sensory         | Touch / body-state input                      |
| Memory         | `#8B5CF6` | internal        | Slower dynamics, persistence                  |
| Emotion        | `#EF4444` | internal        | Valence / arousal modulation                  |
| Association    | `#F59E0B` | integrative hub | Cross-modal binding, boosted by co-activation |

Each zone is defined by a `ZoneSpec` (name, color, normalized seed position in the unit cube, modality flag) in `DEFAULT_ZONES`.

## Zone Assignment Rule

For a grid of side `grid_size`, every neuron's integer coordinate is normalized into the unit cube and assigned to the zone whose **seed position is nearest** (Euclidean):

```
zone(neuron) = argmin_z  || normalized_position(neuron) − seed(z) ||
```

This Voronoi tessellation produces contiguous territories around each seed. Because seeds are spread through the cube, the regions tile space naturally; changing the number or position of seeds re-parcellates the whole volume without touching any other code.

You can inspect exact per-zone counts at runtime:

```python
from positronic_brain.zones import get_zone_info
info = get_zone_info(grid_size=4)
for z in info:
    print(z.name, z.count)
```

## Why a Voronoi Layout?

- **Generality**: works for any number of zones and any grid size — no hand-written `if` cascade per grid.
- **Contiguity**: each zone is a single connected blob, which (combined with distance-biased wiring) means intra-zone synapses dominate and inter-zone fibers are rarer and longer — exactly the cortical motif we want.
- **Hub placement**: placing the Association seed near the center makes it adjacent to several sensory regions, so convergent cross-modal fibers terminate there naturally.

## Visual Encoding in the App

- **Color**: each zone has a distinct, dark-theme-friendly hue (see table above). Inhibitory neurons can additionally be highlighted because their E/I type is a fixed buffer.
- **Size + opacity**: both increase with a neuron's instantaneous firing rate. At rest most neurons are small and semi-transparent; strongly driven neurons "pop".
- **Connections**: drawn only for the strongest active synapses, **split between excitatory (warm) and inhibitory (cool)** so that the ~4× inhibitory magnitude scaling does not visually dominate. Many strong fibers are intra-zone or between adjacent zones; long-range fibers appear when the Association zone is recruited.

## Experiment Ideas Using Zone Layout

1. **Sensory competition**: drive Visual at 0.9 and Auditory at 0.1, then slowly raise Auditory. Watch shared inhibitory circuitry partially suppress the visual region.
2. **Affective tagging**: co-drive Somatosensory + Emotion (the "painful touch" preset) and compare the readout to Somatosensory alone.
3. **Association as amplifier**: set Visual + Auditory moderate and Association high. The output rises more than the sum of the sensory drives because the hub recruits additional pathways.

## Extending the Zoning

To explore different parcellations, edit `DEFAULT_ZONES` (add/remove/reposition seeds) or pass a custom list of `ZoneSpec` objects to `assign_zones`. Because the connectivity generator only cares about 3D positions, you can implement arbitrary zone maps without touching any other module. Update `BrainConfig.zone_names` accordingly so the readout input dimension matches.

The visualizer and the input scatter logic automatically pick up the `zones` buffer and color/route neurons accordingly.
