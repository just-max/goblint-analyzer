open MyCFG
open Queue
open Cil
include CompareAST
open DiffLib

let f = Printf.sprintf

let eq_node (x, fun1) (y, fun2) =
  match x,y with
  | Statement s1, Statement s2 -> eq_stmt ~cfg_comp:true (s1, fun1) (s2, fun2)
  | Function f1, Function f2 -> eq_varinfo f1.svar f2.svar
  | FunctionEntry f1, FunctionEntry f2 -> eq_varinfo f1.svar f2.svar
  | _ -> false

(* TODO: compare ASMs properly instead of simply always assuming that they are not the same *)
let eq_edge x y = match x, y with
  | Assign (lv1, rv1), Assign (lv2, rv2) -> eq_lval lv1 lv2 && eq_exp rv1 rv2
  | Proc (None,f1,ars1), Proc (None,f2,ars2) -> eq_exp f1 f2 && GobList.equal eq_exp ars1 ars2
  | Proc (Some r1,f1,ars1), Proc (Some r2,f2,ars2) ->
    eq_lval r1 r2 && eq_exp f1 f2 && GobList.equal eq_exp ars1 ars2
  | Entry f1, Entry f2 -> eq_varinfo f1.svar f2.svar
  | Ret (None,fd1), Ret (None,fd2) -> eq_varinfo fd1.svar fd2.svar
  | Ret (Some r1,fd1), Ret (Some r2,fd2) -> eq_exp r1 r2 && eq_varinfo fd1.svar fd2.svar
  | Test (p1,b1), Test (p2,b2) -> eq_exp p1 p2 && b1 = b2
  | ASM _, ASM _ -> false
  | Skip, Skip -> true
  | VDecl v1, VDecl v2 -> eq_varinfo v1 v2
  | _ -> false

(* The order of the edges in the list is relevant. Therefore compare them one to one without sorting first *)
let eq_edge_list xs ys = GobList.equal eq_edge xs ys

let to_edge_list : (location * edge) list -> edge list = List.map snd

module MyNodeHashMap = Hashtbl.Make(Node)
type biDirectionNodeMap = {node1to2: node MyNodeHashMap.t; node2to1: node MyNodeHashMap.t}

module MakeSet (K : Set.OrderedType) = struct
  include Set.Make (K)
  let to_list s = List.of_seq (to_seq s)
end

module MakeMap (K : Map.OrderedType) = struct
  include Map.Make (K)
  let of_list l = of_seq (List.to_seq l)
  let to_list m = List.of_seq (to_seq m)
  let diff m1 m2 = merge (fun _ v1 v2 -> match v1, v2 with Some x, None -> Some x | _ -> None) m1 m2
end

module ListExtensions = struct
  open List
  let (let*) xs f = concat_map f xs
  let (let+) xs f = map f xs

  let[@tail_mod_cons] rec take n = function x :: xs when n > 0 -> x :: take (n - 1) xs | _ -> []

  let enumerate xs = List.mapi (fun i x -> i, x) xs

  let sort_on ?(compare = Stdlib.compare) k xs =
    map (fun x -> k x, x) xs
    |> sort (fun (k1, _) (k2, _) -> compare k1 k2)
    |> map (fun (_, x) -> x)

  let maximum_on ?(compare = Stdlib.compare) k xs =
    let rec impl k1 x = function
      | [] -> x
      | (k2, y) :: xs -> (if compare k1 k2 >= 0 then impl k1 x else impl k2 y) xs
    in match List.map (fun x -> k x, x) xs with
    | [] -> failwith "empty list"
    | (kx, x) :: xs -> impl kx x xs
end


type ('m, 'q) matching = {
  old_to_new : 'm ;
  new_to_old : 'm ;
  queue : 'q ;
}

type ('n, 'e) digraph = 'n -> ('e * 'n) list

let str_list str_element xs = "[" ^ String.concat "; " (List.map str_element xs) ^ "]"
let str_tuple str_element_a str_element_b (a, b) = Printf.sprintf "(%s, %s)" (str_element_a a) (str_element_b b)

type 'a record_member_printer = P : string * ('a -> 'm) * ('m -> string) -> 'a record_member_printer
let str_record (str_members : 'a record_member_printer list) (r : 'a) =
  "{" ^ String.concat " ; " (List.map (fun (P (name, value, str)) -> name ^ " = " ^ str (value r)) str_members) ^ "}"


let cmp_tuple cmp_element_a cmp_element_b ((a1, b1) : 'a * 'b) ((a2, b2) : 'a * 'b) =
  match cmp_element_a a1 a2 with 0 -> cmp_element_b b1 b2 | n -> n

let trace_ to_string x = print_endline (to_string x); x

let compare_digraphs_nondet (type n e)
    ?(n_string = Fun.const "<node>") ?(e_string = Fun.const "<edge>")
    ?(limit = 8)
    (n_compare : n -> n -> int) (* both arguments in same graph (old/new) *)
    (* (e_compare : e -> e -> int) *) (* both arguments in same graph (old/new) *)
    (en_match : e -> n -> e -> n -> bool) (* edge in old, node in old, edge in new, node in new *)
    (graph_old : (n, e) digraph) (graph_new : (n, e) digraph)
    (start_old : n) (start_new : n) =
  (* let n_map = (module MakeMap (struct type t = n let compare = n_compare end) : Map.S) in *)
  let module NOrd = struct type t = n let compare = n_compare end in
  let module NMap = MakeMap (NOrd) in
  let module NSet = MakeSet (NOrd) in
  let module NPairSet = MakeSet (struct type t = n * n let compare = cmp_tuple n_compare n_compare end) in

  (* let list_of_nmap m = List.of_seq @@ NMap.to_seq m in *)
  (* let list_of_nset s = List.of_seq @@ NSet.to_seq s in *)
  let str_nmap value m = "{" ^ String.concat "; " (List.map (fun (n, v) -> n_string n ^ ": " ^ value v) @@ NMap.to_list m) ^ "}" in
  let str_nset s = "{" ^ String.concat "; " (List.map n_string @@ NSet.to_list s) ^ "}" in
  let str_npairset s = "{" ^ String.concat "; " (List.map (str_tuple n_string n_string) @@ NPairSet.to_list s) ^ "}" in

  let open ListExtensions in

  (* for each successor of n_old, which successors of n_new could it match with?
     runtime: O(m1 * (log m1 + m2 * log m2)) *)
  let find_successor_matches old_to_new new_to_old n_old n_new =
    let outs graph n m = graph n |> List.filter (fun (_, n') -> not (NMap.mem n' m)) in
    let outs_old = outs graph_old n_old old_to_new in
    let outs_new = outs graph_new n_new new_to_old in
    outs_old |> List.map (fun (e_old, n'_old) ->
      n'_old ,
      outs_new |> List.filter_map (fun (e_new, n'_new) ->
        if en_match e_old n'_old e_new n'_new then Some n'_new else None
      ) |> NSet.of_list
    ) |> NMap.of_list
  in

  (* transform the result of [find_successor_matches] into a list of all potential matchings *)
  let make_successor_matches (* old_to_new new_to_old *) succ_matches =
    NMap.fold (fun n'_old ns'_new matchings ->
        let* (old_to_new, new_to_old) as matching = matchings in
        matching :: (
          NSet.to_list ns'_new
          |> List.filter (fun n'_new ->
              not @@ NMap.mem n'_new new_to_old)
          |> List.map (fun n'_new ->
              NMap.add n'_old n'_new old_to_new, NMap.add n'_new n'_old new_to_old)))
      succ_matches
      [NMap.empty, NMap.empty]
  in

  let out1 = str_nmap str_nset in
  let out2 = str_list (str_tuple (str_nmap n_string) (str_nmap n_string)) in
  let out_matching = (str_record
    [P ("old_to_new", (fun r -> r.old_to_new), str_nmap n_string) ;
    P ("new_to_old", (fun r -> r.new_to_old), str_nmap n_string) ;
    P ("queue", (fun r -> r.queue), str_npairset) ;]) in
  let out3 = str_list out_matching in

  (* take a list of matches and extend the BFS represented by each one *)
  let step matchings =
    let* _mi, ({ old_to_new ; new_to_old ; queue } as _m : (n NMap.t, NPairSet.t) matching) = enumerate matchings in
    (* print_endline @@ Printf.sprintf "## matching index: %d" _mi ;
    print_endline @@ out_matching _m ; *)
    let+ extension_matching = NPairSet.fold (fun (n_old, n_new) matchings' ->
        (* print_endline @@ Printf.sprintf "### for the node pair: %s, %s" (n_string n_old) (n_string n_new) ; *)
        (* try and extend the existing match, splitting in case of a conflict *)
        let matches_by_node = find_successor_matches old_to_new new_to_old n_old n_new in
        (* print_endline "#### potential matches by node:" ; print_endline @@ out1 matches_by_node ; *)
        let extensions = make_successor_matches matches_by_node in
        (* print_endline "#### list of alternative matches:" ; print_endline @@ out2 extensions ; *)
        let* extension, _ = extensions in
        (* let* extension, _ = trace_ out2 @@ make_successor_matches @@ (trace_ out1 @@ find_successor_matches old_to_new new_to_old n_old n_new) in *)
        let* (matching' : (n NMap.t, NPairSet.t) matching) = matchings' in
          (* fold in the matches from the extension *)
          NMap.fold (fun n_old n_new ms ->
              let* m = ms in
              let add_pair m = {
                old_to_new = NMap.add n_old n_new m.old_to_new ;
                new_to_old = NMap.add n_new n_old m.new_to_old ;
                queue = NPairSet.add (n_old, n_new) m.queue }
              in
              match NMap.find_opt n_old m.old_to_new, NMap.mem n_new m.new_to_old with (* (NMap.mem n_old m.old_to_new || NMap.mem n_new m.new_to_old) &&  *)
                | None, false -> [add_pair m]
                | Some n'_new, true when n_compare n'_new n'_new = 0 -> [m]
                | _ ->
                  (* conflict: split in two, once unchanged and once with the conflicting node removed and replaced *)
                  let conflicting_pair = n_old, NMap.find n_old m.old_to_new in
                  [m ; add_pair { old_to_new = NMap.remove n_old m.old_to_new ; new_to_old = NMap.remove n_new m.new_to_old ; queue = NPairSet.remove conflicting_pair m.queue }]
              )
            extension
            [matching'])
      queue
      [{ old_to_new = NMap.empty ; new_to_old = NMap.empty ; queue = NPairSet.empty }] (* accumulator: all the ways the extension has been applied, hopefully this list stays short *)
    in
    { old_to_new = NMap.union (fun _ -> failwith "old_to_new: overlapping match") old_to_new extension_matching.old_to_new ;
      new_to_old = NMap.union (fun _ -> failwith "new_to_old: overlapping match") new_to_old extension_matching.new_to_old ;
      queue = extension_matching.queue }
  in

  let score matching = NMap.cardinal matching.old_to_new in

  (* sort on the number of found pairs, descending. TODO: could be done in O(n) rather than O(nlogn) *)
  let cull matchings = ListExtensions.sort_on (fun m -> - score m) matchings |> take limit in

  let rec go i matchings =
    (* print_endline @@ Printf.sprintf "# RUN %d" i ;
    print_endline "current matchings:" ;
    print_endline @@ out3 matchings ;
    print_endline @@ Printf.sprintf "matchings being considered: %d" (List.length matchings); *)
    let step_result = step matchings |> cull in
    if List.exists (fun m -> not @@ NPairSet.is_empty m.queue) step_result
      then go (i + 1) step_result
      else ((* print_endline "# FINISHED"; print_endline @@ out3 step_result; *) step_result)
  in

  let finish matchings =
    (* print_endline "## Final Result"; *)
    (ListExtensions.maximum_on score matchings).old_to_new
    |> trace_ (str_nmap n_string)
    |> NMap.bindings
  in

  let old_to_new = (NMap.singleton start_old start_new) in
  let new_to_old = (NMap.singleton start_new start_old) in
  finish @@ go 0 [{ old_to_new ; new_to_old ; queue = NPairSet.singleton (start_old, start_new) }]

let cfg1 ?(offset = 0) (nid, _) = (match nid - offset with
  | 5 -> [("a", (3, "b")); ("a", (4, "b"))]
  | 3 -> [("t", (2, "c"))]
  | 4 -> [("f", (2, "c"))]
  | 2 -> [("d", (1, "e"))]
  | 1 -> []
  | _ -> failwith ("bad id: " ^ string_of_int nid))
  |> List.map (fun (e_lbl, (k, n_lbl)) -> e_lbl, (k + offset, n_lbl))

let foo () = compare_digraphs_nondet
  (* ~n_string:(fun (nid, lbl) -> Printf.sprintf "(%d, %S)" nid lbl)
  ~e_string:(Printf.sprintf "%S") *)
  ~n_string:(fun (nid, lbl) -> string_of_int nid)
  ~e_string:Fun.id
  compare
  (fun e1 (_, lbl1) e2 (_, lbl2) -> e1 = e2 && lbl1 = lbl2)
  cfg1 (cfg1 ~offset:10)
  (5, "") (15, "")

let digraph_of_cfg (cfg : cfg) n =
  List.map (fun (es, n) -> to_edge_list es, n) (cfg n)

(* This function compares two CFGs by doing a breadth-first search on the old CFG. Matching node tuples are stored in same,
 * nodes from the old CFG for which no matching node can be found are added to diff. For each matching node tuple
 * (fromNode1, fromNode2) found, one iterates over the successors of fromNode1 from the old CFG and checks for a matching node
 * in the succesors of fromNode2 in the new CFG. Matching node tuples are added to the waitingList to repeat the matching
 * process on their successors. If a node from the old CFG can not be matched, it is added to diff and no further
 * comparison is done for its successors. The two function entry nodes make up the tuple to start the comparison from. *)
let compareCfgs (module CfgOld : CfgForward) (module CfgNew : CfgForward) (fun1 : fundec) (fun2 : fundec) =
  let diff = MyNodeHashMap.create 113 in
  let same = {node1to2=MyNodeHashMap.create 113; node2to1=MyNodeHashMap.create 113} in
  let waitingList : (node * node) t = Queue.create () in
(* 
  prerr_endline "cfg1";
  fprint_fundec_dot (module CfgOld) fun1 stderr;
  prerr_endline "cfg2";
  fprint_fundec_dot (module CfgOld) fun2 stderr; *)

  (* let module NoExtraNodeStyles =
    struct
      let defaultNodeStyles = []
      let extraNodeStyles node = []
    end
  in

  (* same as what is used (indirectly) for producing cfg output in files *)
  ignore @@ CfgTools.(fprint_dot (module CfgPrinters (NoExtraNodeStyles)) stderr); *)

  (* fprint_hash_dot stderr (FunctionEntry fun1);
  fprint_hash_dot stderr fun2; *)


(* 
  prerr_endline "= RETURN ========";
  prerr_endline @@ String.concat "\n" @@ List.map Node.show (cfgLeaves (module CfgOld) fun1);
  prerr_endline "================="; *)
  
  let rec compareNext () =
    if Queue.is_empty waitingList then ()
    else
      let fromNode1, fromNode2 = Queue.take waitingList in
      let outList1 = CfgOld.next fromNode1 in
      let outList2 = CfgNew.next fromNode2 in

      (* Find a matching edge and successor node for (edgeList1, toNode1) in the list of successors of fromNode2.
       * If successful, add the matching node tuple to same, else add toNode1 to the differing nodes. *)
      let findMatch (edgeList1, toNode1) =
        let rec aux = function
          | [] -> MyNodeHashMap.replace diff toNode1 ()
          | (locEdgeList2, toNode2)::remSuc' ->
            let edgeList2 = to_edge_list locEdgeList2 in
            (* TODO: don't allow pseudo return node to be equal to normal return node, could make function unchanged, but have different sallstmts *)
            if eq_node (toNode1, fun1) (toNode2, fun2) && eq_edge_list edgeList1 edgeList2 then
              begin
                (* hier beide checks machen *)
                match MyNodeHashMap.find_opt same.node1to2 toNode1 with
                | Some n2 -> if not (Node.equal n2 toNode2) then MyNodeHashMap.replace diff toNode1 () (* should be removed from same set *)
                | None -> MyNodeHashMap.replace same.node1to2 toNode1 toNode2; MyNodeHashMap.replace same.node2to1 toNode2 toNode1; Queue.add (toNode1, toNode2) waitingList
              end
            else aux remSuc' in
        aux outList2 in
      (* For a toNode1 from the list of successors of fromNode1, check whether it might have duplicate matches.
       * In that case declare toNode1 as differing node. Else, try finding a match in the list of successors
       * of fromNode2 in the new CFG using findMatch. *)
      let iterOuts ((locEdgeList1 : edges), toNode1) =
        let edgeList1 = to_edge_list locEdgeList1 in
        (*  *)
        (* Differentiate between a possibly duplicate Test(1,false) edge and a single occurrence. In the first
         * case the edge is directly added to the diff set to avoid undetected ambiguities during the recursive
         * call. *)
        let testFalseEdge edge = match edge with
          | Test (p,b) -> p = Cil.one && b = false
          | _ -> false in
        let posAmbigEdge edgeList =
          let findTestFalseEdge (ll,_) = testFalseEdge (snd (List.hd ll)) in
          let numDuplicates l = List.length (List.find_all findTestFalseEdge l) in
          testFalseEdge (List.hd edgeList) && (numDuplicates outList1 > 1 || numDuplicates outList2 > 1) in
        if posAmbigEdge edgeList1 then MyNodeHashMap.replace diff toNode1 ()
        else findMatch (edgeList1, toNode1) in
      List.iter iterOuts outList1; compareNext () in

  let entryNode1, entryNode2 = (FunctionEntry fun1, FunctionEntry fun2) in
  MyNodeHashMap.replace same.node1to2 entryNode1 entryNode2; MyNodeHashMap.replace same.node2to1 entryNode2 entryNode1;
  Queue.push (entryNode1,entryNode2) waitingList;
  compareNext ();
  (same, diff)
(*  *)


(* let compareCfgsFromLeaves (module CfgOld : CfgBidir) (module CfgNew : CfgBidir) (fun1 : fundec) (fun2 : fundec) =
  let leaves1, leaves2 = cfgLeaves (module CfgOld) fun1, cfgLeaves (module CfgNew) fun2
  in () *)

(* This is the second phase of the CFG comparison of functions. It removes the nodes from the matching node set 'same'
 * that have an incoming backedge in the new CFG that can be reached from a differing new node. This is important to
 * recognize new dependencies between unknowns that are not contained in the infl from the previous run. *)
let reexamine f1 f2 (same : biDirectionNodeMap) (diffNodes1 : unit MyNodeHashMap.t) (module CfgOld : CfgForward) (module CfgNew : CfgBidir) =
  let rec repeat () =
    let check_all_nodes_in_same ps n =
      match List.find_opt (fun p -> not (MyNodeHashMap.mem same.node2to1 p)) ps with
      | None -> true
      | Some p ->
        begin
          let n1 = MyNodeHashMap.find same.node2to1 n in
          MyNodeHashMap.replace diffNodes1 n1 ();
          MyNodeHashMap.remove same.node1to2 n1; MyNodeHashMap.remove same.node2to1 n;
          false
        end in
    (* check if any of the predecessors of n2 in the new CFG are not in the same set *)
    let cond n2 = Node.equal n2 (FunctionEntry f2) || check_all_nodes_in_same (List.map snd (CfgNew.prev n2)) n2 in
    let forall = MyNodeHashMap.fold (fun n2 n1 acc -> acc && cond n2) same.node2to1 true in
    if not forall then repeat () in
  (* remove nodes in the diff set from the same set,
     which can happen if a node in the old CFG is matched with multiple nodes in the new CFG *)
  MyNodeHashMap.iter (fun node1 () ->
      MyNodeHashMap.find_opt same.node1to2 node1
      |> Option.iter (fun node2 -> MyNodeHashMap.remove same.node1to2 node1; MyNodeHashMap.remove same.node2to1 node2))
    diffNodes1;
  repeat ();
  MyNodeHashMap.to_seq same.node1to2, MyNodeHashMap.to_seq_keys diffNodes1

let h_for_all p ht = MyNodeHashMap.fold (fun k v a -> a && p k v) ht true

let checks1 same diff =
  let check name cond = if not cond then failwith ("assertion failed: " ^ name) in

  let one_entry_per_key ht = h_for_all (fun k _ -> List.length (MyNodeHashMap.find_all ht k) <= 1) ht in

  check "one entry per key (node1to2)" (one_entry_per_key same.node1to2);
  check "one entry per key (node2to1)" (one_entry_per_key same.node2to1);
  check "one entry per key (diff)" (one_entry_per_key diff);

  check "mappings are inverse (node1to2/node2to1)"
  @@ h_for_all
    (fun node1 node2 ->
      MyNodeHashMap.find_opt same.node2to1 node2
      |> Option.map (Node.equal node1)
      |> Option.value ~default:false)
    same.node1to2

let compareFun (module CfgOld : CfgBidir) (module CfgNew : CfgBidir) fun1 fun2 =
  let open CfgTools in
  prerr_endline "==================================================================";
  (* fprint_fundec_compare (module CfgNew) [] [] fun1 stderr ; *)
  let same, (diff : unit MyNodeHashMap.t) =
    Stats.time "compare-phase1" (fun () -> compareCfgs (module CfgOld) (module CfgNew) fun1 fun2) () in

    (* todo: hide behind debug (see schema) *)
  checks1 same diff;

  (* if MyNodeHashMap.length diff > 0 then
    fprint_fundec_compare
      (module CfgNew)
      (List.of_seq @@ MyNodeHashMap.to_seq same.node1to2)
      (List.of_seq @@ MyNodeHashMap.to_seq_keys diff)
      fun2 stderr ; *)
  let unchanged, diffNodes1 = Stats.time "compare-phase2" (fun () -> reexamine fun1 fun2 same diff (module CfgOld) (module CfgNew)) () in
  let uc, d = List.of_seq unchanged, List.of_seq diffNodes1 in
  (* Messages.trace_ "compareFun" "%a" ; *)
  (* prerr_endline @@ "unchanged: " ^ (String.concat "; " @@ List.map Node.(fun (a, b) -> Printf.sprintf "%s, %s (%b)" (show_cfg a) (show_cfg b) (a == b)) uc);
  prerr_endline @@ "differences: " ^ (String.concat "; " @@ List.map Node.show_cfg d); *)
  (* if not (GobList.null d) then fprint_fundec_compare (module CfgNew) uc d fun2 stderr; *)
  uc, d

let compare_cfgs_nondet
    ?limit
    (module CfgOld : CfgForward) (module CfgNew : CfgForward)
    fun_old fun_new start_old start_new
    (* (fun_old : fundec) (fun_new : fundec)
    (start_old : node) (start_new : node) *)=
  compare_digraphs_nondet (* type n: node, type e: edge list *)
    ~n_string:Node.show_id
    ?limit
    Node.compare
    (fun e1 n1 e2 n2 -> eq_node (n1, fun_old) (n2, fun_new) && eq_edge_list e1 e2)
    (digraph_of_cfg CfgOld.next)
    (digraph_of_cfg CfgNew.next)
    start_old start_new

module CfgForwardOfBackward (Cfg : CfgForward) : CfgBackward = struct let prev = Cfg.next end
module CfgBackwardOfForward (Cfg : CfgBackward) : CfgForward = struct let next = Cfg.prev end
module MirrorCfg (Cfg : CfgBidir) : CfgBidir = struct
  include CfgForwardOfBackward (Cfg)
  include CfgBackwardOfForward (Cfg)
end

let compare_fun (module CfgOld : CfgBidir) (module CfgNew : CfgBidir) fun1 fun2 =

  print_endline ("start comparing " ^ fun1.svar.vname);

  CfgTools.fprint_fundec_compare_old (module CfgOld) [] [] [] fun1 stdout;
  CfgTools.fprint_fundec_compare_old (module CfgNew) [] [] [] fun2 stdout;

  let unchanged =
    compare_cfgs_nondet
      (module CfgOld)
      (module CfgNew)
      fun1 fun2
      (FunctionEntry fun1) (FunctionEntry fun2)
  in

  (* temporary hack to run the existing reexamine *)
  let open Batteries in (* % *)
  let unchanged_hm =
    {
      node1to2 = unchanged |> List.to_seq |> MyNodeHashMap.of_seq ;
      node2to1 = unchanged |> List.to_seq |> Seq.map (fun (o, n) -> n, o) |> MyNodeHashMap.of_seq ;
    }
  in
  let diff_hm =
    unchanged
    |> List.to_seq
    |> Seq.map fst
    |> Seq.concat_map (CfgOld.next %> List.to_seq %> Seq.map snd)
    |> Seq.filter (not % MyNodeHashMap.mem unchanged_hm.node1to2)
    |> Seq.map (fun o -> o, ())
    |> MyNodeHashMap.of_seq
  in

  let unchanged, diffNodes1 = reexamine fun1 fun2 unchanged_hm diff_hm (module CfgOld) (module CfgNew) in

  (* TODO: use the result of forward search to limit reverse search *)
  let matches_rev =
    compare_cfgs_nondet
      (module (MirrorCfg (CfgOld)))
      (module (MirrorCfg (CfgNew)))
      fun1 fun2
      (Function fun1) (Function fun2)
  in

  let fuzzy =
    matches_rev
    |> List.filter
         MyNodeHashMap.(
           not % fun (f_old, f_new) ->
           mem unchanged_hm.node1to2 f_old || mem unchanged_hm.node2to1 f_new)
  in

  print_endline ("# Fuzzy matches\n" ^ str_list Node.(str_tuple show_id show_id) fuzzy);

  (* let str_npairset s = "[" ^ String.concat "; " (List.map (str_tuple n_string n_string) @@ NPairSet.to_list s) ^ "]" in *)

  print_endline ("done comparing " ^ fun1.svar.vname);

  List.of_seq unchanged, fuzzy, List.of_seq diffNodes1
  (* let same, diff = Stats.time "compare-phase1" (fun () -> compareCfgs (module CfgOld) (module CfgNew) fun1 fun2) () in
  let unchanged, diffNodes1 = Stats.time "compare-phase2" (fun () -> reexamine fun1 fun2 same diff (module CfgOld) (module CfgNew)) () in
  List.of_seq unchanged, List.of_seq diffNodes1 *)

(* idea: sort by source order??? *)

let linearize_digraph (type n e)
    (* ?(n_string = Fun.const "<node>") ?(e_string = Fun.const "<edge>") *)
    (module N : Hashtbl.HashedType with type t = n)
    (graph : (n, e) digraph) (start : n) : (n * e list) list =

  let module HashtblN = Hashtbl.Make (N) in
  let visited : unit HashtblN.t = HashtblN.create 101 in

  let rec go (n : n) (acc : (n * e list) list) : (n * e list) list =

    if HashtblN.mem visited n
      then acc
      else begin
        HashtblN.replace visited n () ;
        let es, ns = graph n |> List.split in
        (* right-fold instead of left-fold is important if order of outbound edges is relevant *)
        (n, es) :: List.fold_right go ns acc
      end

  in go start []

let compare_fun (module CfgOld : CfgBidir) (module CfgNew : CfgBidir) fun_old fun_new =
  let open Batteries in let open Stdlib in
  let linearize_cfg cfg fundec = (* type n = Node.t, type e = edge list *)
    linearize_digraph (module Node) (digraph_of_cfg cfg) (FunctionEntry fundec) in

  let lin_old = linearize_cfg CfgOld.next fun_old in
  let lin_new = linearize_cfg CfgNew.next fun_new in

  let nes_equal (n1, es1) (n2, es2) =
    eq_node (n1, fun_old) (n2, fun_new)
    (* && List.compare_lengths es1 es2 = 0 *)
    && List.equal eq_edge_list es1 es2
  in
  let diff = DiffLib.myers nes_equal lin_old lin_new in
  let u_diff = DiffLib.unify lin_old lin_new diff in

  print_endline @@ "diff for " ^ fun_old.svar.vname;

  List.iter
    (fun (ud : (node * edge list list, node * edge list list) unified_operation) ->
      ud |> DiffLib.show_unified_operation'
        (fun (n, (ess : edge list list)) ->
          f"%-98s %s"
            (ess |> List.map (List.map (Edge.pretty () %> Pretty.sprint ~width:max_int) %> String.concat " | ") |> String.concat {| /\ |})
            (n |> Node.pretty_plain () |> Pretty.sprint ~width:max_int) |> Str.global_replace (Str.regexp "\n[ \t]*") " ")
      |> print_endline)
    u_diff ;

  (* let _matching = List.filter_map (function UUnchanged ((o, _), (n, _)) -> Some (o, n) | _ -> None) u_diff in *)
  let matches, fuzzyMatches, diffNodes1 = compare_fun (module CfgOld) (module CfgNew) fun_old fun_new in
  matches, fuzzyMatches, diffNodes1, u_diff




(* let linearize_digraph (type n e)
    (* ?(n_string = Fun.const "<node>") ?(e_string = Fun.const "<edge>") *)
    (module N : Hashtbl.HashedType with type t = n)
    (graph : (n, e) digraph) (start : n) =

  let module LL = BatDllist in
  let concat onto = List.fold_left (fun o (h, t) -> LL.splice o h; t) onto in

  let module HashtblN = Hashtbl.Make (N) in
  let visited : unit HashtblN.t = HashtblN.create 101 in

  let rec go ?(e : e option) (n : n) =

    let lln = LL.create (e, n) in
    let visited_n = HashtblN.mem visited n in
    HashtblN.replace visited n () ;

    if visited_n
      then lln, lln
      else
        let lines = graph n |> List.rev |> List.map (fun (e, n) -> go ~e n) in
        lln, concat lln (List.rev lines)

  (* TODO:
    - circular doubly linked list, so storing head and tail is not necessary
    - could probably use a list accumulator instead *)
  in go start |> fst |> LL.to_list *)









(* let compare_cfgs_nondet
    (module CfgOld : CfgForward) (module CfgNew : CfgForward)
    fun_old fun_new start_old start_new
    (* (fun_old : fundec) (fun_new : fundec)
    (start_old : node) (start_new : node) *) =
    compare_digraphs_nondet (* type n: node, type e: edge list *)
    Node.compare
    (* Edge.compare *)
    (fun e1 n1 e2 n2 -> eq_node (n1, fun_old) (n2, fun_new) && eq_edge_list e1 e2)
    (digraph_of_cfg CfgOld.next)
    (digraph_of_cfg CfgNew.next)
    start_old start_new *)

  (*         let* (old_to_new, new_to_old) as matching = matchings in
        matching :: (
          let* n'_new = NSet.to_list ns'_new in
          if not @@ NMap.mem n'_new new_to_old
            then [NMap.add n'_old n'_new old_to_new, NMap.add n'_new n'_old new_to_old] else [])) *)

(*         matchings |> List.concat_map (fun ((old_to_new, new_to_old) as matching) ->
          matching ::
          List.filter_map (fun n'_new ->
              if NMap.mem n'_new new_to_old
                then None
                else Some (NMap.add n'_old n'_new old_to_new, NMap.add n'_new n'_old new_to_old))
            (NSet.to_list ns'_new))) *)



(* module IntMap = MakeMap (struct type t = int let compare = compare end)
module NodeMap = MakeMap (struct type t = node let compare = Node.compare end) *)


(* Find leaves (nodes with no successors) in CFG using a DFS. *)
(* let cfgLeaves (module Cfg : CfgForward) (cfgFun : fundec) : node list =
  let seen = MyNodeHashMap.create 113 in
  let rec impl acc = function
    | [] -> []
    | node::nodes ->
      begin
        match Cfg.next node with
        | [] -> node::acc (* leaf node *)
        | outList ->
          let nodes' = List.map snd outList in
          List.iter (fun n -> MyNodeHashMap.replace seen n ()) nodes';
          impl acc (List.filter (fun n -> not @@ MyNodeHashMap.mem seen n) nodes' @ nodes)
      end
  in
  let entryNode = FunctionEntry cfgFun in
  MyNodeHashMap.replace seen entryNode ();
  impl [] [ entryNode ] *)


(* let fprint_hash_dot out cfg =
  let open CfgTools in
  let module NoExtraNodeStyles =
  struct
    let defaultNodeStyles = []
    let extraNodeStyles node = []
  end
  in
  let iter_edges f = H.iter (fun n es -> List.iter (f n) es) cfg in
  fprint_dot (module CfgPrinters (NoExtraNodeStyles)) iter_edges out *)
(* 
let fprint_fundec_dot (module Cfg : CfgBidir) (* live *) fd out =
  let open CfgTools in
  let module ExtraNodeStyles =
  struct
    let defaultNodeStyles = [] (* ["id=\"\\N\""; "URL=\"javascript:show_info('\\N');\""; "style=filled"; "fillcolor=white"] (* \N is graphviz special for node ID *) *)

    let extraNodeStyles _ = []
      (* if live n then
        []
      else
        ["fillcolor=orange"] *)
  end
  in
  let iter_edges = iter_fd_edges (module Cfg) fd in
  fprint_dot (module CfgPrinters (ExtraNodeStyles)) iter_edges out *)