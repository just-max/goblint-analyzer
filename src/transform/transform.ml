open Prelude
open GoblintCil
module M = Messages

(* The GADT-based approach of Query.t is overkill here *)
type queries = {
  ask : ?node:Node.t -> Cil.location -> Queries.ask ;
  must_be_dead : stmt -> bool ;
  must_be_uncalled : fundec -> bool ;
}

module type S = sig
  val transform : queries -> file -> unit (* modifications are done in-place by CIL :( *)
  val name : string
  val requires_file_output : bool
end

let h = Hashtbl.create 13
let register (module T : S) = Hashtbl.replace h T.name (module T : S)

let run_transformations ?(file_output = true) file names ask =
  let active_transformations =
    List.filter_map
    (fun name ->
      match Hashtbl.find_option h name with
      | Some t -> Some (name, t)
      | None -> M.warn_noloc "Transformation %s does not exist!" name; None)
    names
  in

  List.iter (fun (name, (module T : S)) -> T.transform ask file) active_transformations;

  if file_output && List.exists (fun (_, (module T : S)) -> T.requires_file_output) active_transformations then
    let assert_filename = GobConfig.get_string "trans.output" in
    let oc = Stdlib.open_out assert_filename in
    dumpFile defaultCilPrinter oc assert_filename file;
    Stdlib.close_out oc

let run file name = run_transformations ~file_output:false file [name]

module PartialEval = struct
  let loc = ref locUnknown (* when we visit an expression, we need the current location -> store at stmts *)
  class visitor (ask: ?node:Node.t -> Cil.location -> Queries.ask) = object
    inherit nopCilVisitor
    method! vstmt s =
      loc := Cilfacade.get_stmtLoc s;
      (* ignore @@ Pretty.printf "Set loc at stmt %a to %a\n" d_stmt s CilType.Location.pretty !loc; *)
      DoChildren
    method! vexpr e =
      let eval e = match Queries.ID.to_int ((ask !loc).Queries.f (Queries.EvalInt e)) with
        | Some i ->
          let e' = integer @@ IntOps.BigIntOps.to_int i in
          ignore @@ Pretty.printf "Replacing non-constant expression %a with %a at %a\n" d_exp e d_exp e' CilType.Location.pretty !loc;
          e'
        | None ->
          ignore @@ Pretty.printf "Can't replace expression %a at %a\n" d_exp e CilType.Location.pretty !loc; e
      in
      match e with
      | Const _ -> SkipChildren
      | _ -> ChangeDoChildrenPost (e, eval)
  end
  let transform q file =
    visitCilFile (new visitor q.ask) file

  let name = "partial"

  let requires_file_output = false
end
let _ = register (module PartialEval)
