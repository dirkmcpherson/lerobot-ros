"""Provenance tags on stored transitions.

Every transition's ``complementary_info`` carries, as plain floats (the replay buffer
stores complementary info as float tensors):

    is_intervention   1.0 while the human was driving, else 0.0 (upstream key)
    source            SOURCE_DEMO / SOURCE_INTERVENTION / SOURCE_POLICY
    takeover_onset    1.0 on the first human step of each takeover, else 0.0
    episode_index     index of the episode within its stream (demos and online
                      episodes are numbered separately; pair it with ``source``)
    step_index        position of the transition in its episode

The online and the offline replay buffer must hold exactly the same complementary_info
keys: lerobot's ``concatenate_batch_transitions`` concatenates key by key and leaves a
key that is missing on one side at half length, misaligned with the batch. The demo
recorder (``record.py``) therefore writes the same columns the actor adds, and the
learner wrapper refuses to start sampling if the key sets differ.
"""

SOURCE_DEMO, SOURCE_INTERVENTION, SOURCE_POLICY = 0, 1, 2
SOURCE_NAMES = {SOURCE_DEMO: "demo", SOURCE_INTERVENTION: "intervention", SOURCE_POLICY: "policy"}

IS_INTERVENTION = "is_intervention"
PROVENANCE_KEYS = (IS_INTERVENTION, "source", "takeover_onset", "episode_index", "step_index")
DISCRETE_PENALTY = "discrete_penalty"
# Column order shared by the actor's transitions and the demo dataset.
COMPLEMENTARY_KEYS = (DISCRETE_PENALTY, *PROVENANCE_KEYS)


def provenance_tags(interventions: list[bool], episode_index: int, demo: bool = False) -> list[dict[str, float]]:
    """Tags for one episode, from the per-step "was the human driving" flags.

    In a demo every step is ``source = demo`` and there is no takeover onset: no policy
    was running, so no steps led up to a takeover.
    """
    tags = []
    previous = False
    for step, human in enumerate(interventions):
        human = bool(human)
        if demo:
            source, onset = SOURCE_DEMO, False
        else:
            source = SOURCE_INTERVENTION if human else SOURCE_POLICY
            onset = human and not previous
        tags.append(
            {
                IS_INTERVENTION: float(human),
                "source": float(source),
                "takeover_onset": float(onset),
                "episode_index": float(episode_index),
                "step_index": float(step),
            }
        )
        previous = human
    return tags


def tag_episode(transitions: list[dict], episode_index: int) -> list[dict]:
    """Add the provenance tags to one episode of actor transitions, in place."""
    flags = [bool((t.get("complementary_info") or {}).get(IS_INTERVENTION, False)) for t in transitions]
    for transition, tags in zip(transitions, provenance_tags(flags, episode_index), strict=True):
        info = dict(transition.get("complementary_info") or {})
        info.update(tags)
        transition["complementary_info"] = info
    return transitions
