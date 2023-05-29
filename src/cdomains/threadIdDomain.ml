(** Thread ID domains. *)

open GoblintCil
open FlagHelper
open BatPervasives

module type S =
sig
  include Printable.S
  include MapDomain.Groupable with type t := t

  val threadinit: varinfo -> multiple:bool -> t
  val is_main: t -> bool
  val is_unique: t -> bool

  (** Overapproximates whether the first TID can be involved in the creation fo the second TID*)
  val may_create: t -> t -> bool

  (** Is the first TID a must parent of the second thread. Always false if the first TID is not unique *)
  val is_must_parent: t -> t -> bool
end

module type Stateless =
sig
  include S

  val threadenter: t -> t
  val threadenter_node: Node.t -> int option -> varinfo -> t
end

module type Stateful =
sig
  include S

  module D: Lattice.S

  val threadenter: t * D.t -> Node.t -> int option -> varinfo -> t list
  val threadspawn: D.t -> t -> D.t

  (** If it is possible to get a list of unique thread create thus far, get it *)
  val created: t -> D.t -> (t list) option
end



(** Type to represent an abstract thread ID. *)
module FunNode: Stateless =
struct
  include
    Printable.Prod
      (CilType.Varinfo) (
      Printable.Option (
        Printable.Prod
          (Node) (
          Printable.Option
            (WrapperFunctionAnalysis0.ThreadCreateUniqueCount)
            (struct let name = "no index" end)))
        (struct let name = "no node" end))

  let show = function
    | (f, Some (n, i)) ->
      f.vname
      ^ "@" ^ (CilType.Location.show (UpdateCil.getLoc n))
      ^ "#" ^ Option.fold ~none:"top" ~some:string_of_int i
    | (f, None) -> f.vname

  include Printable.SimpleShow (
    struct
      type nonrec t = t
      let show = show
    end
  )

  let threadinit v ~multiple: t = (v, None)

  let threadenter (v, ni : t) : t =
    (v, if GobConfig.get_bool "ana.thread.include-node" then ni else None)

  let threadenter_node n i v = threadenter (v, Some (n, i))

  let is_main = function
    | ({vname; _}, None) -> List.mem vname @@ GobConfig.get_string_list "mainfun"
    | _ -> false

  let is_unique _ = false (* TODO: should this consider main unique? *)
  let may_create _ _ = true
  let is_must_parent _ _ = false
end


module Unit (Base: Stateless): Stateful =
struct
  include Base

  module D = Lattice.Unit

  let threadenter _ n i v = [threadenter_node n i v]
  let threadspawn () _ = ()

  let created _ _ = None
end

module History (Base: Stateless): Stateful =
struct
  module P =
  struct
    include Printable.Liszt (Base)
    (* Prefix is stored in reversed order (main is last) since prepending is more efficient. *)
    let name () = "prefix"
  end
  module S =
  struct
    include SetDomain.Make (Base)
    let name () = "set"
  end
  include Printable.Prod (P) (S)

  let pretty () (p, s) =
    let p = List.rev p in (* show in "unreversed" order *)
    if S.is_empty s then
      P.pretty () p (* hide empty set *)
    else
      Pretty.dprintf "%a, %a" P.pretty p S.pretty s

  let show x = GobPretty.sprint pretty x

  module D =
  struct
    include S
    let name () = "created"
  end

  let is_unique (_, s) =
    S.is_empty s

  let is_must_parent (p,s) (p',s') =
    if not (S.is_empty s) then
      false
    else
      let cdef_ancestor = P.common_suffix p p' in
      P.equal p cdef_ancestor

  let may_create (p,s) (p',s') =
    S.subset (S.union (S.of_list p) s) (S.union (S.of_list p') s')

  let compose ((p, s) as current) ni =
    if BatList.mem_cmp Base.compare ni p then (
      let shared, unique = BatList.span (not % Base.equal ni) p in
      (List.tl unique, S.of_list shared |> S.union s |> S.add ni)
    )
    else if is_unique current then
      (ni :: p, s)
    else
      (p, S.add ni s)

  let threadinit v ~multiple =
    let base_tid = Base.threadinit v ~multiple in
    if multiple then
      ([], S.singleton base_tid)
    else
      ([base_tid], S.empty ())

  let threadenter ((p, _) as current, cs) n i v =
    let ni = Base.threadenter_node n i v in
    let composed = compose current ni in
    if is_unique composed && S.mem ni cs then
      [(p, S.singleton ni); composed] (* also respawn unique version of the thread to keep it reachable while thread ID sets refer to it *)
    else
      [composed]

  let created current cs =
    let els = D.elements cs in
    Some (List.map (compose current) els)

  let threadspawn (cs : D.t) (p, s : t) : D.t =
    (if not (D.is_empty s) then D.elements s else BatList.take 1 p)
    |> List.map Base.threadenter |> D.of_list |> D.union cs

  let is_main = function
    | ([fl], s) when S.is_empty s && Base.is_main fl -> true
    | _ -> false
end

module ThreadLiftNames = struct
  let bot_name = "Bot Threads"
  let top_name = "Top Threads"
end
module Lift (Thread: S) =
struct
  include Lattice.Flat (Thread) (ThreadLiftNames)
  let name () = "Thread"
end

module FlagConfiguredTID:Stateful =
struct
  (* Thread IDs with prefix-set history *)
  module H = History(FunNode)
  (* Plain thread IDs *)
  module P = Unit(FunNode)

  let msg = "FlagConfiguredTID received a value where not exactly one component is set"
  include GroupableFlagHelper(H)(P)(struct
      let msg = msg
      let name = "FlagConfiguredTID"
    end)

  module D = Lattice.Lift2(H.D)(P.D)(struct let bot_name = "bot" let top_name = "top" end)

  let history_enabled () =
    match GobConfig.get_string "ana.thread.domain" with
    | "plain" -> false
    | "history" -> true
    | s -> failwith @@ "Illegal value " ^ s ^ " for ana.thread.domain"

  let threadinit v ~multiple =
    if history_enabled () then
      (Some (H.threadinit v ~multiple), None)
    else
      (None, Some (P.threadinit v ~multiple))

  let is_main = unop H.is_main P.is_main
  let is_unique = unop H.is_unique P.is_unique
  let may_create = binop H.may_create P.may_create
  let is_must_parent = binop H.is_must_parent P.is_must_parent

  let created x d =
    let lifth x' d' =
      let hres = H.created x' d' in
      match hres with
      | None -> None
      | Some l -> Some (List.map (fun x -> (Some x, None)) l)
    in
    let liftp x' d' =
      let pres = P.created x' d' in
      match pres with
      | None -> None
      | Some l -> Some (List.map (fun x -> (None, Some x)) l)
    in
    match x, d with
    | (Some x', None), `Lifted1 d' -> lifth x' d'
    | (Some x', None), `Bot -> lifth x' (H.D.bot ())
    | (Some x', None), `Top -> lifth x' (H.D.top ())
    | (None, Some x'), `Lifted2 d' -> liftp x' d'
    | (None, Some x'), `Bot -> liftp x' (P.D.bot ())
    | (None, Some x'), `Top -> liftp x' (P.D.top ())
    | _ -> None

  let threadenter x n i v =
    match x with
    | ((Some x', None), `Lifted1 d) -> H.threadenter (x',d) n i v |> List.map (fun t -> (Some t, None))
    | ((Some x', None), `Bot) -> H.threadenter (x',H.D.bot ()) n i v |> List.map (fun t -> (Some t, None))
    | ((Some x', None), `Top) -> H.threadenter (x',H.D.top ()) n i v |> List.map (fun t -> (Some t, None))
    | ((None, Some x'), `Lifted2 d) -> P.threadenter (x',d) n i v |> List.map (fun t -> (None, Some t))
    | ((None, Some x'), `Bot) -> P.threadenter (x',P.D.bot ()) n i v |> List.map (fun t -> (None, Some t))
    | ((None, Some x'), `Top) -> P.threadenter (x',P.D.top ()) n i v |> List.map (fun t -> (None, Some t))
    | _ -> failwith msg

  let threadspawn (x : D.t) (n : t) : D.t =
    match x, n with
    | `Lifted1 x', (Some n', None) -> `Lifted1 (H.threadspawn x' n')
    | `Lifted2 x', (None, Some n') -> `Lifted2 (P.threadspawn x' n')
    | `Bot, (Some n', None) when history_enabled () -> `Lifted1 (H.threadspawn (H.D.bot ()) n')
    | `Bot, (None, Some n')                         -> `Lifted2 (P.threadspawn (P.D.bot ()) n')
    | `Top, (Some n', None) when history_enabled () -> `Lifted1 (H.threadspawn (H.D.top ()) n')
    | `Top, (None, Some n')                         -> `Lifted2 (P.threadspawn (P.D.top ()) n')
    | _ -> failwith msg

  let name () = "FlagConfiguredTID: " ^ if history_enabled () then H.name () else P.name ()
end

module Thread = FlagConfiguredTID

module ThreadLifted = Lift (Thread)
