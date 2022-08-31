open Cil
open MyCFG
open CompareTypes
include CompareAST
include CompareCFG

let should_reanalyze (fdec: Cil.fundec) =
  List.mem fdec.svar.vname (GobConfig.get_string_list "incremental.force-reanalyze.funs")

(* If some CFGs of the two functions to be compared are provided, a fine-grained CFG comparison is done that also determines which
 * nodes of the function changed. If on the other hand no CFGs are provided, the "old" AST comparison on the CIL.file is
 * used for functions. Then no information is collected regarding which parts/nodes of the function changed. *)
let eqF (a: Cil.fundec) (b: Cil.fundec) (cfgs : ((cfg * cfg) * (cfg * cfg)) option) =
  let unchangedHeader = eq_varinfo a.svar b.svar && GobList.equal eq_varinfo a.sformals b.sformals in
  let identical, diffOpt =
    if should_reanalyze a then
      false, None
    else
      let sameDef = unchangedHeader && GobList.equal eq_varinfo a.slocals b.slocals in
      if not sameDef then
        (false, None)
      else
        match cfgs with
        | None -> eq_block (a.sbody, a) (b.sbody, b), None
        | Some ((cfgOld, cfgOldBack), (cfgNew, cfgNewBack)) ->
          let matches, fuzzyMatches, diffNodes1, diff =
            compare_fun
              (CfgTools.makeCFGBidir cfgOld cfgOldBack)
              (CfgTools.makeCFGBidir cfgNew cfgNewBack)
              a b
          in
          if diffNodes1 = [] then (true, None)
          else (false, Some {
            unchangedNodes = matches; fuzzyMatchNodes = fuzzyMatches; diff = diff; primObsoleteNodes = diffNodes1;
            (* cfg_old = { forward = cfgOld; backward = cfgOldBack; };
            cfg_new = { forward = cfgNew; backward = cfgNewBack; } *)})
  in
  identical, unchangedHeader, diffOpt

let eq_glob (a: global) (b: global) (cfgs : ((cfg * cfg) * (cfg * cfg)) option) = match a, b with
  | GFun (f,_), GFun (g,_) -> (* print_endline (Printf.sprintf "%s %s" f.svar.vname g.svar.vname);*) eqF f g cfgs
  | GVar (x, init_x, _), GVar (y, init_y, _) -> eq_varinfo x y, false, None (* ignore the init_info - a changed init of a global will lead to a different start state *)
  | GVarDecl (x, _), GVarDecl (y, _) -> eq_varinfo x y, false, None
  | _ -> ignore @@ Pretty.printf "Not comparable: %a and %a\n" Cil.d_global a Cil.d_global b; false, false, None

let compareCilFiles ?(eq=eq_glob) (oldAST: file) (newAST: file) =
  (* old, oldBack, new, newBack *)
  let cfgs = if GobConfig.get_string "incremental.compare" = "cfg"
    then Some (CfgTools.getCFG oldAST, CfgTools.getCFG newAST)
    else None in

  let addGlobal map global  =
    try
      let gid = identifier_of_global global in
      let gid_to_string gid = match gid.global_t with
        | Var -> "Var " ^ gid.name
        | Decl -> "Decl " ^ gid.name
        | Fun -> "Fun " ^ gid.name in
      if GlobalMap.mem gid map then failwith ("Duplicate global identifier: " ^ gid_to_string gid) else GlobalMap.add gid global map
    with
      Not_found -> map
  in
  let changes = empty_change_info () in
  global_typ_acc := [];
  let checkUnchanged map global =
    try
      let ident = identifier_of_global global in
      let old_global = GlobalMap.find ident map in
      (* Do a (recursive) equal comparison ignoring location information *)
      let identical, unchangedHeader, diff = eq old_global global cfgs in
      if identical
      then changes.unchanged <- {current = global; old = old_global} :: changes.unchanged
      else changes.changed <- {current = global; old = old_global; unchangedHeader; diff} :: changes.changed
    with Not_found -> () (* Global was no variable or function, it does not belong into the map *)
  in
  let checkExists map global =
    match identifier_of_global global with
    | name -> GlobalMap.mem name map
    | exception Not_found -> true (* return true, so isn't considered a change *)
  in
  (* Store a map from functionNames in the old file to the function definition*)
  let oldMap = Cil.foldGlobals oldAST addGlobal GlobalMap.empty in
  let newMap = Cil.foldGlobals newAST addGlobal GlobalMap.empty in
  (*  For each function in the new file, check whether a function with the same name
      already existed in the old version, and whether it is the same function. *)
  Cil.iterGlobals newAST
    (fun glob -> checkUnchanged oldMap glob);

  (* We check whether functions have been added or removed *)
  Cil.iterGlobals newAST (fun glob -> if not (checkExists oldMap glob) then changes.added <- (glob::changes.added));
  Cil.iterGlobals oldAST (fun glob -> if not (checkExists newMap glob) then changes.removed <- (glob::changes.removed));
  changes

(** Given an (optional) equality function between [Cil.global]s, an old and a new [Cil.file], this function computes a [change_info],
    which describes which [global]s are changed, unchanged, removed and added.  *)
let compareCilFiles ?eq (oldAST: file) (newAST: file) =
  Stats.time "compareCilFiles" (compareCilFiles ?eq oldAST) newAST
