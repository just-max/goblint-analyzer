(** Abstract domains representing sets. *)
open Pretty

(* Exception raised when the set domain can not support the requested operation.
 * This will be raised, when trying to iterate a set that has been set to Top *)
exception Unsupported of string

let unsupported s = raise (Unsupported s)

(** A set domain must support all the standard library set operations.
    They have been copied instead of included since our [empty] has a different signature. *)
module type S =
sig
  include Lattice.S
  type elt
  val empty: unit -> t
  val is_empty: t -> bool
  val mem: elt -> t -> bool
  val add: elt -> t -> t
  val singleton: elt -> t
  val remove: elt -> t -> t
  val union: t -> t -> t
  val inter: t -> t -> t
  val diff: t -> t -> t
  val subset: t -> t -> bool
  val disjoint: t -> t -> bool
  val iter: (elt -> unit) -> t -> unit
  val map: (elt -> elt) -> t -> t
  val fold: (elt -> 'a -> 'a) -> t -> 'a -> 'a
  val for_all: (elt -> bool) -> t -> bool
  val exists: (elt -> bool) -> t -> bool
  val filter: (elt -> bool) -> t -> t
  val partition: (elt -> bool) -> t -> t * t
  val cardinal: t -> int
  val elements: t -> elt list
  val of_list: elt list -> t
  val min_elt: t -> elt
  val max_elt: t -> elt
  val choose: t -> elt
end


(** A functor for creating a simple set domain, there is no top element, and
  * calling [top ()] will raise an exception *)
module Make (Base: Printable.S): S with
  type elt = Base.t and
  type t = BatSet.Make (Base).t = (* TODO: remove, only needed in VarEq for some reason... *)
struct
  include Printable.Blank
  include BatSet.Make(Base)
  let name () = "Set (" ^ Base.name () ^ ")"
  let empty _ = empty
  let leq  = subset
  let join = union
  let widen = join
  let meet = inter
  let narrow = meet
  let bot = empty
  let is_bot = is_empty
  let top () = unsupported "Make.top"
  let is_top _ = false

  let map f s =
    let add_to_it x s = add (f x) s in
    fold add_to_it s (empty ())

  let pretty () x =
    let elts = elements x in
    let content = List.map (Base.pretty ()) elts in
    let rec separate x =
      match x with
      | [] -> []
      | [x] -> [x]
      | (x::xs) -> x ++ (text ", ") :: separate xs
    in
    let separated = separate content in
    let content = List.fold_left (++) nil separated in
    (text "{") ++ content ++ (text "}")

  (** Short summary for sets. *)
  let show x : string =
    let all_elems : string list = List.map Base.show (elements x) in
    Printable.get_short_list "{" "}" all_elems

  let to_yojson x = [%to_yojson: Base.t list] (elements x)

  let equal x y =
    cardinal x = cardinal y
    && for_all (fun e -> exists (Base.equal e) y) x

  let hash x = fold (fun x y -> y + Base.hash x) x 0

  let pretty_diff () ((x:t),(y:t)): Pretty.doc =
    if leq x y then dprintf "%s: These are fine!" (name ()) else
    if is_bot y then dprintf "%s: %a instead of bot" (name ()) pretty x else begin
      let evil = choose (diff x y) in
      Pretty.dprintf "%s: %a not leq %a\n  @[because %a@]" (name ()) pretty x pretty y Base.pretty evil
    end
  let printXml f xs =
    BatPrintf.fprintf f "<value>\n<set>\n";
    iter (Base.printXml f) xs;
    BatPrintf.fprintf f "</set>\n</value>\n"

  let arbitrary () = QCheck.map ~rev:elements of_list @@ QCheck.small_list (Base.arbitrary ())
end

(** A functor for creating a path sensitive set domain, that joins the base
  * analysis whenever the user elements coincide. Just as above there is no top
  * element, and calling [top ()] will raise an exception *)
(* TODO: unused *)
module SensitiveConf (C: Printable.ProdConfiguration) (Base: Lattice.S) (User: Printable.S) =
struct
  module Elt = Printable.ProdConf (C) (Base) (User)
  include Make(Elt)
  let name () = "Sensitive " ^ name ()

  let leq s1 s2 =
    (* I want to check that forall e in x, the same key is in y with it's base
     * domain element being leq of this one *)
    let p (b1,u1) = exists (fun (b2,u2) -> User.equal u1 u2 && Base.leq b1 b2) s2 in
    for_all p s1

  let pretty_diff () ((x:t),(y:t)): Pretty.doc =
    Pretty.dprintf "%s: %a not leq %a" (name ()) pretty x pretty y

  let join s1 s2 =
    (* Ok, so for each element (b2,u2) in s2, we check in s1 for elements that have
     * equal user values (there should be at most 1) and we either join with it, or
     * just add the element to our accumulator res and remove it from s1 *)
    let f (b2,u2) (s1,res) =
      let (s1_match, s1_rest) = partition (fun (b1,u1) -> User.equal u1 u2) s1 in
      let el =
        try let (b1,u1) = choose s1_match in (Base.join b1 b2, u2)
        with Not_found -> (b2,u2)
      in
      (s1_rest, add el res)
    in
    let (s1', res) = fold f s2 (s1, empty ()) in
    union s1' res

  let add e s = join (singleton e) s

  (* The meet operation is slightly different from the above, I think this is
   * the right thing, the intuition is from thinking of this as a MapBot *)
  let meet s1 s2 =
    let f (b2,u2) (s1,res) =
      let (s1_match, s1_rest) = partition (fun (b1,u1) -> User.equal u1 u2) s1 in
      let res =
        try
          let (b1,u1) = choose s1_match in
          add (Base.meet b1 b2, u2) res
        with Not_found -> res
      in
      (s1_rest, res)
    in
    snd (fold f s2 (s1, empty ()))
end

(* TODO: unused *)
module Sensitive = SensitiveConf (struct
    let expand_fst = true
    let expand_snd = true
  end)

(** Auxiliary signature for naming the top element *)
module type ToppedSetNames =
sig
  val topname: string
end

module LiftTop (S: S) (N: ToppedSetNames): S with
  type elt = S.elt and
  type t = [`Top | `Lifted of S.t] = (* Expose t for HoareDomain.Set_LiftTop *)
struct
  include Printable.Std

  include Lattice.LiftTop (S)

  type elt = S.elt

  let empty () = `Lifted (S.empty ())
  let is_empty x =
    match x with
    | `Top -> false
    | `Lifted x -> S.is_empty x
  let mem x s =
    match s with
    | `Top -> true
    | `Lifted s -> S.mem x s
  let add x s =
    match s with
    | `Top -> `Top
    | `Lifted s -> `Lifted (S.add x s)
  let singleton x = `Lifted (S.singleton x)
  let remove x s =
    match s with
    | `Top -> `Top   (* NB! NB! NB! *)
    | `Lifted s -> `Lifted (S.remove x s)
  let union x y =
    match x, y with
    | `Top, _ -> `Top
    | _, `Top -> `Top
    | `Lifted x, `Lifted y -> `Lifted (S.union x y)
  let inter x y =
    match x, y with
    | `Top, y -> y
    | x, `Top -> x
    | `Lifted x, `Lifted y -> `Lifted (S.inter x y)
  let diff x y =
    match x, y with
    | x, `Top -> empty ()
    | `Top, y -> `Top (* NB! NB! NB! *)
    | `Lifted x, `Lifted y -> `Lifted (S.diff x y)
  let subset x y =
    match x, y with
    | _, `Top -> true
    | `Top, _ -> false
    | `Lifted x, `Lifted y -> S.subset x y
  let disjoint x y =
    match x, y with
    | `Top, `Top -> false
    | `Lifted x, `Top
    | `Top, `Lifted x -> S.is_empty x
    | `Lifted x, `Lifted y -> S.disjoint x y

  let schema normal abnormal x =
    match x with
    | `Top -> unsupported abnormal
    | `Lifted t -> normal t
  let schema_default v f = function
    | `Top -> v
    | `Lifted x -> f x
  (* HACK! Map is an exception in that it doesn't throw an exception! *)
  let map f x =
    match x with
    | `Top -> `Top
    | `Lifted t -> `Lifted (S.map f t)

  let iter f = schema (S.iter f) "iter on `Top"
  (*  let map f = schema (fun t -> `Lifted (S.map f t)) "map"*)
  let fold f x e = schema (fun t -> S.fold f t e) "fold on `Top" x
  let for_all f = schema_default false (S.for_all f)
  let exists f = schema_default true (S.exists f)
  let filter f = schema (fun t -> `Lifted (S.filter f t)) "filter on `Top"
  let elements = schema S.elements "elements on `Top"
  let of_list xs = `Lifted (S.of_list xs)
  let cardinal = schema S.cardinal "cardinal on `Top"
  let min_elt = schema S.min_elt "min_elt on `Top"
  let max_elt = schema S.max_elt "max_elt on `Top"
  let choose = schema S.choose "choose on `Top"
  let partition f = schema (fun t -> match S.partition f t
                             with (a,b) -> (`Lifted a, `Lifted b)) "filter on `Top"


  (* The printable implementation *)
  (* Overrides `Top text *)

  let pretty () x =
    match x with
    | `Top -> text N.topname
    | `Lifted t -> S.pretty () t

  let show x : string =
    match x with
    | `Top -> N.topname
    | `Lifted t -> S.show t


  (* Lattice implementation *)
  (* Lift separately because lattice order might be different from subset order, e.g. after Reverse *)

  let bot () = `Lifted (S.bot ())
  let is_bot x =
    match x with
    | `Top -> false
    | `Lifted x -> S.is_bot x

  let leq x y =
    match x, y with
    | _, `Top -> true
    | `Top, _ -> false
    | `Lifted x, `Lifted y -> S.leq x y
  let join x y =
    match x, y with
    | `Top, _ -> `Top
    | _, `Top -> `Top
    | `Lifted x, `Lifted y -> `Lifted (S.join x y)
  let widen x y = (* assumes y to be bigger than x *)
    match x, y with
    | `Top, _
    | _, `Top -> `Top
    | `Lifted x, `Lifted y -> `Lifted (S.widen x y)
  let meet x y =
    match x, y with
    | `Top, y -> y
    | x, `Top -> x
    | `Lifted x, `Lifted y -> `Lifted (S.meet x y)
  let narrow x y =
    match x, y with
    | `Top, y -> y
    | x, `Top -> x
    | `Lifted x, `Lifted y -> `Lifted (S.narrow x y)

  let invariant c = function
    | `Top -> Invariant.none
    | `Lifted s -> S.invariant c s

  let arbitrary () = QCheck.set_print show (arbitrary ())
end

(** Functor for creating artificially topped set domains. *)
module ToppedSet (Base: Printable.S) (N: ToppedSetNames): S with
  type elt = Base.t and
  type t = [`Top | `Lifted of Make (Base).t] = (* TODO: don't expose t *)
struct
  module S = Make (Base)
  include LiftTop (S) (N)
end

(* This one just removes the extra "{" notation and also by always returning
 * false for the isSimple, the answer looks better, but this is essentially a
 * hack. All the pretty printing needs some rethinking. *)
module HeadlessSet (Base: Printable.S) =
struct
  include Make(Base)

  let name () = "Headless " ^ name ()
  let pretty () x =
    let elts = elements x in
    let content = List.map (Base.pretty ()) elts in
    let rec separate x =
      match x with
      | [] -> []
      | [x] -> [x]
      | (x::xs) -> x ++ (text ", ") ++ line :: separate xs
    in
    let separated = separate content in
    let content = List.fold_left (++) nil separated in
    content

  let pretty_diff () ((x:t),(y:t)): Pretty.doc =
    Pretty.dprintf "%s: %a not leq %a" (name ()) pretty x pretty y
  let printXml f xs =
    iter (Base.printXml f) xs
end

(** Reverses lattice order of a set domain while keeping the set operations same. *)
module Reverse (Base: S) =
struct
  include Base
  include Lattice.Reverse (Base)
end

module type FiniteSetElems =
sig
  type t
  val elems: t list
end

module FiniteSet (E:Printable.S) (Elems:FiniteSetElems with type t = E.t) =
struct
  module E =
  struct
    include E
    let arbitrary () = QCheck.oneofl Elems.elems
  end

  include Make (E)
  let top () = of_list Elems.elems
  let is_top x = equal x (top ())
end
