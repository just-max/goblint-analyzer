open Cil
open MyCFG

type cfg_bidir = {
  forward: cfg;
  backward: cfg
}


type nodes_diff = {
  unchangedNodes: (node * node) list;
  fuzzyMatchNodes: (node * node) list;
  primObsoleteNodes: node list; (** primary obsolete nodes -> all obsolete nodes are reachable from these *)
}

type unchanged_global = {
  old: global;
  current: global
}
(** For semantically unchanged globals, still keep old and current version of global for resetting current to old. *)

type changed_global = {
  old: global;
  current: global;
  unchangedHeader: bool;
  diff: nodes_diff option
}

type change_info = {
  mutable changed: changed_global list;
  mutable unchanged: unchanged_global list;
  mutable removed: global list;
  mutable added: global list
}

let empty_change_info () : change_info = {added = []; removed = []; changed = []; unchanged = []}
