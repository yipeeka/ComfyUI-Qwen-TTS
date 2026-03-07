# ComfyUI-Qwen-TTS Utility Nodes
# General-purpose list manipulation helpers.


# ─────────────────────────────────────────────────────────────────────────────
# Node: Append Any To List
#
# Appends a single item of *any* type to a list and returns the extended list.
#
# Typical wiring pattern (chain to build a list incrementally):
#
#   [NodeA] ──item──► [AppendAnyToList] ──list_out──► [AppendAnyToList] ──list_out──► ...
#                        ▲                                  ▲
#                   (no list_in)                       list_in from previous
#
# Or to feed an existing OUTPUT_IS_LIST node's output:
#
#   [SomeListNode] ──list_out──► list_in ─┐
#   [NodeB]        ──────────── item   ──►[AppendAnyToList] ──list_out──► downstream
#
# ─────────────────────────────────────────────────────────────────────────────
class AppendAnyToListNode:
    """
    Appends one item (any type) to an existing list, or starts a new list.

    - list_in  : optional; must come from a node with OUTPUT_IS_LIST=(True,)
                 or from another AppendAnyToList. If omitted, a new list is created.
    - item     : the value to append (any ComfyUI type).
    - list_out : the resulting list (OUTPUT_IS_LIST=True so downstream nodes
                 with INPUT_IS_LIST=True receive all elements).

    Both INPUT_IS_LIST and OUTPUT_IS_LIST are True so ComfyUI never iterates
    list_in element-by-element across repeated node executions.
    """

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "item": ("*", {"forceInput": True, "tooltip": "Item to append (any type)"}),
            },
            "optional": {
                "list_in": ("*", {
                    "forceInput": True,
                    "tooltip": "Existing list to append to (leave disconnected to start a new list)",
                }),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("list_out",)
    FUNCTION = "append_item"
    CATEGORY = "Qwen3-TTS/Utils"
    DESCRIPTION = (
        "Append any single item to a list. "
        "Chain multiple nodes to build lists of arbitrary length. "
        "list_out uses OUTPUT_IS_LIST=True."
    )

    def append_item(self, item: list, list_in: list = None):
        # With INPUT_IS_LIST=True:
        #   item    = [actual_value]  (single item wrapped in list by ComfyUI)
        #   list_in = [elem1, elem2, ...]  (the full accumulated list)
        existing = list(list_in) if list_in else []
        existing.extend(item)   # item is already [val], so this appends exactly one element
        return (existing,)


# ─────────────────────────────────────────────────────────────────────────────
# Exported mappings (consumed by __init__.py)
# ─────────────────────────────────────────────────────────────────────────────
UTILS_NODE_CLASS_MAPPINGS = {
    "FB_AppendAnyToList": AppendAnyToListNode,
}

UTILS_NODE_DISPLAY_NAME_MAPPINGS = {
    "FB_AppendAnyToList": "📋 Append Any To List",
}
