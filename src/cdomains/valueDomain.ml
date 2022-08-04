open Cil
open Pretty
open GobConfig
open PrecisionUtil

include PreValueDomain
module Offs = Lval.Offset (IndexDomain)
module M = Messages
module GU = Goblintutil
module Expp = ExpDomain
module Q = Queries
module BI = IntOps.BigIntOps
module AddrSetDomain = SetDomain.ToppedSet(Addr)(struct let topname = "All" end)
module ArrIdxDomain = IndexDomain

module type S =
sig
  include Lattice.S
  type offs
  val eval_offset: Q.ask -> (AD.t -> t) -> t-> offs -> exp option -> lval option -> typ -> t
  val update_offset: Q.ask -> t -> offs -> t -> exp option -> lval -> typ -> t
  val update_array_lengths: (exp -> t) -> t -> Cil.typ -> t
  val affect_move: ?replace_with_const:bool -> Q.ask -> t -> varinfo -> (exp -> int option) -> t
  val affecting_vars: t -> varinfo list
  val invalidate_value: Q.ask -> typ -> t -> t
  val is_safe_cast: typ -> typ -> bool
  val cast: ?torg:typ -> typ -> t -> t
  val smart_join: (exp -> BI.t option) -> (exp -> BI.t option) -> t -> t ->  t
  val smart_widen: (exp -> BI.t option) -> (exp -> BI.t option) ->  t -> t -> t
  val smart_leq: (exp -> BI.t option) -> (exp -> BI.t option) -> t -> t -> bool
  val is_immediate_type: typ -> bool
  val bot_value: typ -> t
  val is_bot_value: t -> bool
  val init_value: typ -> t
  val top_value: typ -> t
  val is_top_value: t -> typ -> bool
  val zero_init_value: typ -> t

  val project: precision -> t -> t
end

module type Blob =
sig
  type value
  type size
  type origin
  include Lattice.S with type t = value * size * origin

  val value: t -> value
  val invalidate_value: Q.ask -> typ -> t -> t
end

(* ZeroInit is true if malloc was used to allocate memory and it's false if calloc was used *)
module ZeroInit = Lattice.Fake(Basetype.RawBools)

module Blob (Value: S) (Size: IntDomain.Z)=
struct
  include Lattice.Prod3 (Value) (Size) (ZeroInit)
  let name () = "blob"
  type value = Value.t
  type size = Size.t
  type origin = ZeroInit.t
  let printXml f (x, y, z) =
    BatPrintf.fprintf f "<value>\n<map>\n<key>\n%s\n</key>\n%a<key>\nsize\n</key>\n%a<key>\norigin\n</key>\n%a</map>\n</value>\n" (XmlUtil.escape (Value.name ())) Value.printXml x Size.printXml y ZeroInit.printXml z

  let value (a, b, c) = a
  let invalidate_value ask t (v, s, o) = Value.invalidate_value ask t v, s, o

  let invariant c (v, _, _) = Value.invariant c v
end

module Threads = ConcDomain.ThreadSet

module rec Compound: S with type t = [
    | `Top
    | `Int of ID.t
    | `Address of AD.t
    | `Struct of Structs.t
    | `Union of Unions.t
    | `Array of CArrays.t
    | `Blob of Blobs.t
    | `Thread of Threads.t
    | `Bot
  ] and type offs = (fieldinfo,IndexDomain.t) Lval.offs =
struct
  type t = [
    | `Top
    | `Int of ID.t
    | `Address of AD.t
    | `Struct of Structs.t
    | `Union of Unions.t
    | `Array of CArrays.t
    | `Blob of Blobs.t
    | `Thread of Threads.t
    | `Bot
  ] [@@deriving eq, ord, hash]

  let is_mutex_type (t: typ): bool = match t with
  | TNamed (info, attr) -> info.tname = "pthread_mutex_t" || info.tname = "spinlock_t"
  | TInt (IInt, attr) -> hasAttribute "mutex" attr
  | _ -> false

  let is_immediate_type t = is_mutex_type t || isFunctionType t

  let is_thread_type = function
    | TNamed ({tname = "pthread_t"; _}, _) -> true
    | _ -> false

  let rec bot_value (t: typ): t =
    match t with
    | TInt _ -> `Bot (*`Int (ID.bot ()) -- should be lower than any int or address*)
    | TPtr _ -> `Address (AD.bot ())
    | TComp ({cstruct=true; _} as ci,_) -> `Struct (Structs.create (fun fd -> bot_value fd.ftype) ci)
    | TComp ({cstruct=false; _},_) -> `Union (Unions.bot ())
    | TArray (ai, None, _) ->
      `Array (CArrays.make (IndexDomain.bot ()) (bot_value ai))
    | TArray (ai, Some exp, _) ->
      let l = BatOption.map Cilint.big_int_of_cilint (Cil.getInteger (Cil.constFold true exp)) in
      `Array (CArrays.make (BatOption.map_default (IndexDomain.of_int (Cilfacade.ptrdiff_ikind ())) (IndexDomain.bot ()) l) (bot_value ai))
    | t when is_thread_type t -> `Thread (ConcDomain.ThreadSet.empty ())
    | TNamed ({ttype=t; _}, _) -> bot_value t
    | _ -> `Bot

  let is_bot_value x =
    match x with
    | `Int x -> ID.is_bot x
    | `Address x -> AD.is_bot x
    | `Struct x -> Structs.is_bot x
    | `Union x -> Unions.is_bot x
    | `Array x -> CArrays.is_bot x
    | `Blob x -> Blobs.is_bot x
    | `Thread x -> Threads.is_bot x
    | `Bot -> true
    | `Top -> false

  let rec init_value (t: typ): t = (* top_value is not used here because structs, blob etc will not contain the right members *)
    match t with
    | t when is_mutex_type t -> `Top
    | TInt (ik,_) -> `Int (ID.top_of ik)
    | TPtr _ -> `Address AD.top_ptr
    | TComp ({cstruct=true; _} as ci,_) -> `Struct (Structs.create (fun fd -> init_value fd.ftype) ci)
    | TComp ({cstruct=false; _},_) -> `Union (Unions.top ())
    | TArray (ai, None, _) ->
      `Array (CArrays.make (IndexDomain.bot ())  (if (get_string "ana.base.arrays.domain"="partitioned" || get_string "ana.base.arrays.domain"="unroll") then (init_value ai) else (bot_value ai)))
    | TArray (ai, Some exp, _) ->
      let l = BatOption.map Cilint.big_int_of_cilint (Cil.getInteger (Cil.constFold true exp)) in
      `Array (CArrays.make (BatOption.map_default (IndexDomain.of_int (Cilfacade.ptrdiff_ikind ())) (IndexDomain.bot ()) l) (if (get_string "ana.base.arrays.domain"="partitioned" || get_string "ana.base.arrays.domain"="unroll") then (init_value ai) else (bot_value ai)))
    (* | t when is_thread_type t -> `Thread (ConcDomain.ThreadSet.empty ()) *)
    | TNamed ({ttype=t; _}, _) -> init_value t
    | _ -> `Top

  let rec top_value (t: typ): t =
    match t with
    | TInt (ik,_) -> `Int (ID.(cast_to ik (top_of ik)))
    | TPtr _ -> `Address AD.top_ptr
    | TComp ({cstruct=true; _} as ci,_) -> `Struct (Structs.create (fun fd -> top_value fd.ftype) ci)
    | TComp ({cstruct=false; _},_) -> `Union (Unions.top ())
    | TArray (ai, None, _) ->
      `Array (CArrays.make (IndexDomain.top ()) (if (get_string "ana.base.arrays.domain"="partitioned" || get_string "ana.base.arrays.domain"="unroll") then (top_value ai) else (bot_value ai)))
    | TArray (ai, Some exp, _) ->
      let l = BatOption.map Cilint.big_int_of_cilint (Cil.getInteger (Cil.constFold true exp)) in
      `Array (CArrays.make (BatOption.map_default (IndexDomain.of_int (Cilfacade.ptrdiff_ikind ())) (IndexDomain.top_of (Cilfacade.ptrdiff_ikind ())) l) (if (get_string "ana.base.arrays.domain"="partitioned" || get_string "ana.base.arrays.domain"="unroll") then (top_value ai) else (bot_value ai)))
    | TNamed ({ttype=t; _}, _) -> top_value t
    | _ -> `Top

  let is_top_value x (t: typ) =
    match x with
    | `Int x -> ID.is_top_of (Cilfacade.get_ikind (t)) x
    | `Address x -> AD.is_top x
    | `Struct x -> Structs.is_top x
    | `Union x -> Unions.is_top x
    | `Array x -> CArrays.is_top x
    | `Blob x -> Blobs.is_top x
    | `Thread x -> Threads.is_top x
    | `Top -> true
    | `Bot -> false

    let rec zero_init_value (t:typ): t =
      match t with
      | TInt (ikind, _) -> `Int (ID.of_int ikind BI.zero)
      | TPtr _ -> `Address AD.null_ptr
      | TComp ({cstruct=true; _} as ci,_) -> `Struct (Structs.create (fun fd -> zero_init_value fd.ftype) ci)
      | TComp ({cstruct=false; _} as ci,_) ->
        let v = try
          (* C99 6.7.8.10: the first named member is initialized (recursively) according to these rules *)
          let firstmember = List.hd ci.cfields in
          `Lifted firstmember, zero_init_value firstmember.ftype
        with
          (* Union with no members ò.O *)
          Failure _ -> Unions.top ()
        in
        `Union(v)
      | TArray (ai, None, _) ->
        `Array (CArrays.make (IndexDomain.top_of (Cilfacade.ptrdiff_ikind ())) (zero_init_value ai))
      | TArray (ai, Some exp, _) ->
        let l = BatOption.map Cilint.big_int_of_cilint (Cil.getInteger (Cil.constFold true exp)) in
        `Array (CArrays.make (BatOption.map_default (IndexDomain.of_int (Cilfacade.ptrdiff_ikind ())) (IndexDomain.top_of (Cilfacade.ptrdiff_ikind ())) l) (zero_init_value ai))
      (* | t when is_thread_type t -> `Thread (ConcDomain.ThreadSet.empty ()) *)
      | TNamed ({ttype=t; _}, _) -> zero_init_value t
      | _ -> `Top

  let tag_name : t -> string = function
    | `Top -> "Top" | `Int _ -> "Int" | `Address _ -> "Address" | `Struct _ -> "Struct" | `Union _ -> "Union" | `Array _ -> "Array" | `Blob _ -> "Blob" | `Thread _ -> "Thread" | `Bot -> "Bot"

  include Printable.Std
  let name () = "compound"

  type offs = (fieldinfo,IndexDomain.t) Lval.offs


  let bot () = `Bot
  let is_bot x = x = `Bot
  let bot_name = "Uninitialized"
  let top () = `Top
  let is_top x = x = `Top
  let top_name = "Unknown"

  let pretty () state =
    match state with
    | `Int n ->  ID.pretty () n
    | `Address n ->  AD.pretty () n
    | `Struct n ->  Structs.pretty () n
    | `Union n ->  Unions.pretty () n
    | `Array n ->  CArrays.pretty () n
    | `Blob n ->  Blobs.pretty () n
    | `Thread n -> Threads.pretty () n
    | `Bot -> text bot_name
    | `Top -> text top_name

  let show state =
    match state with
    | `Int n ->  ID.show n
    | `Address n ->  AD.show n
    | `Struct n ->  Structs.show n
    | `Union n ->  Unions.show n
    | `Array n ->  CArrays.show n
    | `Blob n ->  Blobs.show n
    | `Thread n -> Threads.show n
    | `Bot -> bot_name
    | `Top -> top_name

  let pretty_diff () (x,y) =
    match (x,y) with
    | (`Int x, `Int y) -> ID.pretty_diff () (x,y)
    | (`Address x, `Address y) -> AD.pretty_diff () (x,y)
    | (`Struct x, `Struct y) -> Structs.pretty_diff () (x,y)
    | (`Union x, `Union y) -> Unions.pretty_diff () (x,y)
    | (`Array x, `Array y) -> CArrays.pretty_diff () (x,y)
    | (`Blob x, `Blob y) -> Blobs.pretty_diff () (x,y)
    | (`Thread x, `Thread y) -> Threads.pretty_diff () (x, y)
    | _ -> dprintf "%s: %a not same type as %a" (name ()) pretty x pretty y

  (************************************************************
   * Functions for getting state out of a compound:
   ************************************************************)

  (* is a cast t1 to t2 invertible, i.e., content-preserving? TODO also use abstract value? *)
  let is_safe_cast t2 t1 = match t2, t1 with
    (*| TPtr _, t -> bitsSizeOf t <= bitsSizeOf !upointType
      | t, TPtr _ -> bitsSizeOf t >= bitsSizeOf !upointType*)
    | TInt (ik,_), TFloat (fk,_) (* does a1 fit into ik's range? *)
    | TFloat (fk,_), TInt (ik,_) (* can a1 be represented as fk? *)
      -> false (* TODO precision *)
    | _ -> IntDomain.Size.is_cast_injective ~from_type:t1 ~to_type:t2 && bitsSizeOf t2 >= bitsSizeOf t1
  (*| _ -> false*)

  let ptr_ikind () = match !upointType with TInt (ik,_) -> ik | _ -> assert false

  exception CastError of string

  let typ_eq t1 t2 = match typeSig t1, typeSig t2 with
    (* f() and f(void) are not the same (1. no args specified, 2. specified as no args), but we don't care for function pointer casts TODO why does CIL have type f(void) for function definitions f(){..}? *)
    | TSFun (r1, None, false, _), TSFun (r2, Some [], false, _)
    | TSFun (r1, Some [], false, _), TSFun (r2, None, false, _)
      -> r1 = r2
    | a, b -> a = b

  let cast_addr t a =
    let rec stripVarLenArr = function
      | TPtr(t, args) -> TPtr(stripVarLenArr t, args)
      | TArray(t, None, args) -> TArray(stripVarLenArr t, None, args)
      | TArray(t, Some exp, args) when isConstant exp -> TArray(stripVarLenArr t, Some exp, args)
      | TArray(t, Some exp, args) -> TArray(stripVarLenArr t, None, args)
      | t -> t
    in
    let rec adjust_offs v o d =
      let ta = try Addr.type_offset v.vtype o with Addr.Type_offset (t,s) -> raise (CastError s) in
      let info = Pretty.(sprint ~width:0 @@ dprintf "Ptr-Cast %a from %a to %a" Addr.pretty (Addr.Addr (v,o)) d_type ta d_type t) in
      M.tracel "casta" "%s\n" info;
      let err s = raise (CastError (s ^ " (" ^ info ^ ")")) in
      match Stdlib.compare (bitsSizeOf (stripVarLenArr t)) (bitsSizeOf (stripVarLenArr ta)) with (* TODO is it enough to compare the size? -> yes? *)
      | 0 ->
        M.tracel "casta" "same size\n";
        if not (typ_eq t ta) then err "Cast to different type of same size."
        else (M.tracel "casta" "SUCCESS!\n"; o)
      | c when c > 0 -> (* cast to bigger/outer type *)
        M.tracel "casta" "cast to bigger size\n";
        if d = Some false then err "Ptr-cast to type of incompatible size!" else
        if o = `NoOffset then err "Ptr-cast to outer type, but no offset to remove."
        else if Addr.is_zero_offset o then adjust_offs v (Addr.remove_offset o) (Some true)
        else err "Ptr-cast to outer type, but possibly from non-zero offset."
      | _ -> (* cast to smaller/inner type *)
        M.tracel "casta" "cast to smaller size\n";
        if d = Some true then err "Ptr-cast to type of incompatible size!" else
          begin match ta, t with
            (* struct to its first field *)
            | TComp ({cfields = fi::_; _}, _), _ ->
              M.tracel "casta" "cast struct to its first field\n";
              adjust_offs v (Addr.add_offsets o (`Field (fi, `NoOffset))) (Some false)
            (* array of the same type but different length, e.g. assign array (with length) to array-ptr (no length) *)
            | TArray (t1, _, _), TArray (t2, _, _) when typ_eq t1 t2 -> o
            (* array to its first element *)
            | TArray _, _ ->
              M.tracel "casta" "cast array to its first element\n";
              adjust_offs v (Addr.add_offsets o (`Index (IndexDomain.of_int (Cilfacade.ptrdiff_ikind ()) BI.zero, `NoOffset))) (Some false)
            | _ -> err @@ "Cast to neither array index nor struct field."
                          ^ Pretty.(sprint ~width:0 @@ dprintf " is_zero_offset: %b" (Addr.is_zero_offset o))
          end
    in
    let one_addr = let open Addr in function
        | Addr ({ vtype = TVoid _; _} as v, offs) when not (Cilfacade.isCharType t) -> (* we had no information about the type (e.g. malloc), so we add it; ignore for casts to char* since they're special conversions (N1570 6.3.2.3.7) *)
          Addr ({ v with vtype = t }, offs) (* HACK: equal varinfo with different type, causes inconsistencies down the line, when we again assume vtype being "right", but joining etc gives no consideration to which type version to keep *)
        | Addr (v, o) as a ->
          begin try Addr (v, (adjust_offs v o None)) (* cast of one address by adjusting the abstract offset *)
            with CastError s -> (* don't know how to handle this cast :( *)
              M.tracel "caste" "%s\n" s;
              a (* probably garbage, but this is deref's problem *)
              (*raise (CastError s)*)
            | SizeOfError (s,t) ->
              M.warn "size of error: %s" s;
              a
          end
        | x -> x (* TODO we should also keep track of the type here *)
    in
    let a' = AD.map one_addr a in
    M.tracel "cast" "cast_addr %a to %a is %a!\n" AD.pretty a d_type t AD.pretty a'; a'

  (* this is called for:
   * 1. normal casts
   * 2. dereferencing pointers (needed?)
  *)
  let cast ?torg t v =
    (*if v = `Bot || (match torg with Some x -> is_safe_cast t x | None -> false) then v else*)
    match v with
    | `Bot
    | `Thread _ ->
      v
    | _ ->
      let log_top (_,l,_,_) = Messages.tracel "cast" "log_top at %d: %a to %a is top!\n" l pretty v d_type t in
      let t = unrollType t in
      let v' = match t with
        | TFloat (fk,_) -> log_top __POS__; `Top
        | TInt (ik,_) ->
          `Int (ID.cast_to ?torg ik (match v with
              | `Int x -> x
              | `Address x when AD.equal x AD.null_ptr -> ID.of_int (ptr_ikind ()) BI.zero
              | `Address x when AD.is_not_null x -> ID.of_excl_list (ptr_ikind ()) [BI.zero]
              (*| `Struct x when Structs.cardinal x > 0 ->
                let some  = List.hd (Structs.keys x) in
                let first = List.hd some.fcomp.cfields in
                (match Structs.get x first with `Int x -> x | _ -> raise CastError)*)
              | _ -> log_top __POS__; ID.top_of ik
            ))
        | TEnum ({ekind=ik; _},_) ->
          `Int (ID.cast_to ?torg ik (match v with
              | `Int x -> (* TODO warn if x is not in the constant values of ei.eitems? (which is totally valid (only ik is relevant for wrapping), but might be unintended) *) x
              | _ -> log_top __POS__; ID.top_of ik
            ))
        | TPtr (t,_) when isVoidType t || isVoidPtrType t ->
          (match v with
          | `Address a -> v
          | `Int i -> `Int(ID.cast_to ?torg (ptr_ikind ()) i)
          | _ -> v (* TODO: Does it make sense to have things here that are neither `Address nor `Int? *)
          )
          (* cast to voidPtr are ignored TODO what happens if our value does not fit? *)
        | TPtr (t,_) ->
          `Address (match v with
              | `Int x when ID.to_int x = Some BI.zero -> AD.null_ptr
              | `Int x -> AD.top_ptr
              (* we ignore casts to void*! TODO report UB! *)
              | `Address x -> (match t with TVoid _ -> x | _ -> cast_addr t x)
              (*| `Address x -> x*)
              | _ -> log_top __POS__; AD.top_ptr
            )
        | TArray (ta, l, _) -> (* TODO, why is the length exp option? *)
          (* TODO handle casts between different sizes? *)
          `Array (match v with
              | `Array x -> x
              | _ -> log_top __POS__; CArrays.top ()
            )
        | TComp (ci,_) -> (* struct/union *)
          (* rather clumsy, but our abstract values don't keep their type *)
          let same_struct x = (* check if both have the same parent *)
            match Structs.keys x, ci.cfields with
            | k :: _, f :: _ -> compFullName k.fcomp = compFullName f.fcomp (* compinfo is cyclic, so we only check the name *)
            | _, _ -> false (* can't say if struct is empty *)
          in
          (* 1. casting between structs of different type does not work
           * 2. dereferencing a casted pointer works, but is undefined behavior because of the strict aliasing rule (compiler assumes that pointers of different type can never point to the same location)
          *)
          if ci.cstruct then
            `Struct (match v with
                | `Struct x when same_struct x -> x
                | `Struct x when ci.cfields <> [] ->
                  let first = List.hd ci.cfields in
                  Structs.(replace (Structs.create (fun fd -> top_value fd.ftype) ci) first (get x first))
                | _ -> log_top __POS__; Structs.create (fun fd -> top_value fd.ftype) ci
              )
          else
            `Union (match v with
                | `Union x (* when same (Unions.keys x) *) -> x
                | _ -> log_top __POS__; Unions.top ()
              )
        (* | _ -> log_top (); `Top *)
        | TVoid _ -> log_top __POS__; `Top
        | TBuiltin_va_list _ ->
          (* cast to __builtin_va_list only happens in preprocessed SV-COMP files where vararg declarations are more explicit *)
          log_top __POS__; `Top
        | _ -> log_top __POS__; assert false
      in
      let s_torg = match torg with Some t -> Prelude.Ana.sprint d_type t | None -> "?" in
      Messages.tracel "cast" "cast %a from %s to %a is %a!\n" pretty v s_torg d_type t pretty v'; v'


  let warn_type op x y =
    if GobConfig.get_bool "dbg.verbose" then
      ignore @@ printf "warn_type %s: incomparable abstr. values %s and %s at %a: %a and %a\n" op (tag_name x) (tag_name y) CilType.Location.pretty !Tracing.current_loc pretty x pretty y

  let rec leq x y =
    match (x,y) with
    | (_, `Top) -> true
    | (`Top, _) -> false
    | (`Bot, _) -> true
    | (_, `Bot) -> false
    | (`Int x, `Int y) -> ID.leq x y
    | (`Int x, `Address y) when ID.to_int x = Some BI.zero && not (AD.is_not_null y) -> true
    | (`Int _, `Address y) when AD.may_be_unknown y -> true
    | (`Address _, `Int y) when ID.is_top_of (Cilfacade.ptrdiff_ikind ()) y -> true
    | (`Address x, `Address y) -> AD.leq x y
    | (`Struct x, `Struct y) -> Structs.leq x y
    | (`Union x, `Union y) -> Unions.leq x y
    | (`Array x, `Array y) -> CArrays.leq x y
    | (`Blob x, `Blob y) -> Blobs.leq x y
    | `Blob (x,s,o), y -> leq (x:t) y
    | x, `Blob (y,s,o) -> leq x (y:t)
    | (`Thread x, `Thread y) -> Threads.leq x y
    | (`Int x, `Thread y) -> true
    | (`Address x, `Thread y) -> true
    | _ -> warn_type "leq" x y; false

  let rec join x y =
    match (x,y) with
    | (`Top, _) -> `Top
    | (_, `Top) -> `Top
    | (`Bot, x) -> x
    | (x, `Bot) -> x
    | (`Int x, `Int y) -> (try `Int (ID.join x y) with IntDomain.IncompatibleIKinds m -> Messages.warn "%s" m; `Top)
    | (`Int x, `Address y)
    | (`Address y, `Int x) -> `Address (match ID.to_int x with
        | Some x when BI.equal x BI.zero -> AD.join AD.null_ptr y
        | Some x -> AD.(join y not_null)
        | None -> AD.join y AD.top_ptr)
    | (`Address x, `Address y) -> `Address (AD.join x y)
    | (`Struct x, `Struct y) -> `Struct (Structs.join x y)
    | (`Union (f,x), `Union (g,y)) -> `Union (match UnionDomain.Field.join f g with
        | `Lifted f -> (`Lifted f, join x y) (* f = g *)
        | x -> (x, `Top)) (* f <> g *)
    | (`Array x, `Array y) -> `Array (CArrays.join x y)
    | (`Blob x, `Blob y) -> `Blob (Blobs.join x y)
    | `Blob (x,s,o), y
    | y, `Blob (x,s,o) -> `Blob (join (x:t) y, s, o)
    | (`Thread x, `Thread y) -> `Thread (Threads.join x y)
    | (`Int x, `Thread y)
    | (`Thread y, `Int x) ->
      `Thread y (* TODO: ignores int! *)
    | (`Address x, `Thread y)
    | (`Thread y, `Address x) ->
      `Thread y (* TODO: ignores address! *)
    | _ ->
      warn_type "join" x y;
      `Top

  let rec smart_join x_eval_int y_eval_int  (x:t) (y:t):t =
    let join_elem: (t -> t -> t) = smart_join x_eval_int y_eval_int in  (* does not compile without type annotation *)
    match (x,y) with
    | (`Top, _) -> `Top
    | (_, `Top) -> `Top
    | (`Bot, x) -> x
    | (x, `Bot) -> x
    | (`Int x, `Int y) -> (try `Int (ID.join x y) with IntDomain.IncompatibleIKinds m -> Messages.warn "%s" m; `Top)
    | (`Int x, `Address y)
    | (`Address y, `Int x) -> `Address (match ID.to_int x with
        | Some x when BI.equal BI.zero x -> AD.join AD.null_ptr y
        | Some x -> AD.(join y not_null)
        | None -> AD.join y AD.top_ptr)
    | (`Address x, `Address y) -> `Address (AD.join x y)
    | (`Struct x, `Struct y) -> `Struct (Structs.join_with_fct join_elem x y)
    | (`Union (f,x), `Union (g,y)) -> `Union (match UnionDomain.Field.join f g with
        | `Lifted f -> (`Lifted f, join_elem x y) (* f = g *)
        | x -> (x, `Top)) (* f <> g *)
    | (`Array x, `Array y) -> `Array (CArrays.smart_join x_eval_int y_eval_int x y)
    | (`Blob x, `Blob y) -> `Blob (Blobs.join x y) (* `Blob can not contain array -> normal join  *)
    | `Blob (x,s,o), y
    | y, `Blob (x,s,o) ->
      `Blob (join (x:t) y, s, o)
    | (`Thread x, `Thread y) -> `Thread (Threads.join x y)
    | (`Int x, `Thread y)
    | (`Thread y, `Int x) ->
      `Thread y (* TODO: ignores int! *)
    | (`Address x, `Thread y)
    | (`Thread y, `Address x) ->
      `Thread y (* TODO: ignores address! *)
    | _ ->
      warn_type "join" x y;
      `Top

  let rec smart_widen x_eval_int y_eval_int x y:t =
    let widen_elem: (t -> t -> t) = smart_widen x_eval_int y_eval_int in (* does not compile without type annotation *)
    match (x,y) with
    | (`Top, _) -> `Top
    | (_, `Top) -> `Top
    | (`Bot, x) -> x
    | (x, `Bot) -> x
    | (`Int x, `Int y) -> (try `Int (ID.widen x y) with IntDomain.IncompatibleIKinds m -> Messages.warn "%s" m; `Top)
    | (`Int x, `Address y)
    | (`Address y, `Int x) -> `Address (match ID.to_int x with
        | Some x when BI.equal BI.zero x -> AD.widen AD.null_ptr y
        | Some x -> AD.(widen y not_null)
        | None -> AD.widen y AD.top_ptr)
    | (`Address x, `Address y) -> `Address (AD.widen x y)
    | (`Struct x, `Struct y) -> `Struct (Structs.widen_with_fct widen_elem x y)
    | (`Union (f,x), `Union (g,y)) -> `Union (match UnionDomain.Field.widen f g with
        | `Lifted f -> `Lifted f, widen_elem x y  (* f = g *)
        | x -> x, `Top) (* f <> g *)
    | (`Array x, `Array y) -> `Array (CArrays.smart_widen x_eval_int y_eval_int x y)
    | (`Blob x, `Blob y) -> `Blob (Blobs.widen x y) (* `Blob can not contain array -> normal widen  *)
    | (`Thread x, `Thread y) -> `Thread (Threads.widen x y)
    | (`Int x, `Thread y)
    | (`Thread y, `Int x) ->
      `Thread y (* TODO: ignores int! *)
    | (`Address x, `Thread y)
    | (`Thread y, `Address x) ->
      `Thread y (* TODO: ignores address! *)
    | _ ->
      warn_type "widen" x y;
      `Top


  let rec smart_leq x_eval_int y_eval_int x y =
    let leq_elem:(t ->t -> bool) = smart_leq x_eval_int y_eval_int in (* does not compile without type annotation *)
    match (x,y) with
    | (_, `Top) -> true
    | (`Top, _) -> false
    | (`Bot, _) -> true
    | (_, `Bot) -> false
    | (`Int x, `Int y) -> ID.leq x y
    | (`Int x, `Address y) when ID.to_int x = Some BI.zero && not (AD.is_not_null y) -> true
    | (`Int _, `Address y) when AD.may_be_unknown y -> true
    | (`Address _, `Int y) when ID.is_top_of (Cilfacade.ptrdiff_ikind ()) y -> true
    | (`Address x, `Address y) -> AD.leq x y
    | (`Struct x, `Struct y) ->
          Structs.leq_with_fct leq_elem x y
    | (`Union (f, x), `Union (g, y)) ->
        UnionDomain.Field.leq f g && leq_elem x y
    | (`Array x, `Array y) -> CArrays.smart_leq x_eval_int y_eval_int x y
    | (`Blob x, `Blob y) -> Blobs.leq x y (* `Blob can not contain array -> normal leq  *)
    | (`Thread x, `Thread y) -> Threads.leq x y
    | (`Int x, `Thread y) -> true
    | (`Address x, `Thread y) -> true
    | _ -> warn_type "leq" x y; false

  let rec meet x y =
    match (x,y) with
    | (`Bot, _) -> `Bot
    | (_, `Bot) -> `Bot
    | (`Top, x) -> x
    | (x, `Top) -> x
    | (`Int x, `Int y) -> `Int (ID.meet x y)
    | (`Int _, `Address _) -> meet x (cast (TInt(ptr_ikind (),[])) y)
    | (`Address x, `Int y) -> `Address (AD.meet x (AD.of_int (module ID:IntDomain.Z with type t = ID.t) y))
    | (`Address x, `Address y) -> `Address (AD.meet x y)
    | (`Struct x, `Struct y) -> `Struct (Structs.meet x y)
    | (`Union x, `Union y) -> `Union (Unions.meet x y)
    | (`Array x, `Array y) -> `Array (CArrays.meet x y)
    | (`Blob x, `Blob y) -> `Blob (Blobs.meet x y)
    | (`Thread x, `Thread y) -> `Thread (Threads.meet x y)
    | (`Int x, `Thread y)
    | (`Thread y, `Int x) ->
      `Int x (* TODO: ignores thread! *)
    | (`Address x, `Thread y)
    | (`Thread y, `Address x) ->
      `Address x (* TODO: ignores thread! *)
    | _ ->
      warn_type "meet" x y;
      `Bot

  let rec widen x y =
    match (x,y) with
    | (`Top, _) -> `Top
    | (_, `Top) -> `Top
    | (`Bot, x) -> x
    | (x, `Bot) -> x
    | (`Int x, `Int y) -> (try `Int (ID.widen x y) with IntDomain.IncompatibleIKinds m -> Messages.warn "%s" m; `Top)
    | (`Int x, `Address y)
    | (`Address y, `Int x) -> `Address (match ID.to_int x with
        | Some x when BI.equal x BI.zero -> AD.widen AD.null_ptr y
        | Some x -> AD.(widen y not_null)
        | None -> AD.widen y AD.top_ptr)
    | (`Address x, `Address y) -> `Address (AD.widen x y)
    | (`Struct x, `Struct y) -> `Struct (Structs.widen x y)
    | (`Union (f,x), `Union (g,y)) -> `Union (match UnionDomain.Field.widen f g with
        | `Lifted f -> (`Lifted f, widen x y) (* f = g *)
        | x -> (x, `Top))
    | (`Array x, `Array y) -> `Array (CArrays.widen x y)
    | (`Blob x, `Blob y) -> `Blob (Blobs.widen x y)
    | (`Thread x, `Thread y) -> `Thread (Threads.widen x y)
    | (`Int x, `Thread y)
    | (`Thread y, `Int x) ->
      `Thread y (* TODO: ignores int! *)
    | (`Address x, `Thread y)
    | (`Thread y, `Address x) ->
      `Thread y (* TODO: ignores address! *)
    | _ ->
      warn_type "widen" x y;
      `Top

  let rec narrow x y =
    match (x,y) with
    | (`Int x, `Int y) -> `Int (ID.narrow x y)
    | (`Int _, `Address _) -> narrow x (cast IntDomain.Size.top_typ y)
    | (`Address x, `Int y) -> `Address (AD.narrow x (AD.of_int (module ID:IntDomain.Z with type t = ID.t) y))
    | (`Address x, `Address y) -> `Address (AD.narrow x y)
    | (`Struct x, `Struct y) -> `Struct (Structs.narrow x y)
    | (`Union x, `Union y) -> `Union (Unions.narrow x y)
    | (`Array x, `Array y) -> `Array (CArrays.narrow x y)
    | (`Blob x, `Blob y) -> `Blob (Blobs.narrow x y)
    | (`Thread x, `Thread y) -> `Thread (Threads.narrow x y)
    | (`Int x, `Thread y)
    | (`Thread y, `Int x) ->
      `Int x (* TODO: ignores thread! *)
    | (`Address x, `Thread y)
    | (`Thread y, `Address x) ->
      `Address x (* TODO: ignores thread! *)
    | x, `Top | `Top, x -> x
    | x, `Bot | `Bot, x -> `Bot
    | _ ->
      warn_type "narrow" x y;
      x

  let rec invalidate_value (ask:Q.ask) typ (state:t) : t =
    let typ = unrollType typ in
    let invalid_struct compinfo old =
      let nstruct = Structs.create (fun fd -> invalidate_value ask fd.ftype (Structs.get old fd)) compinfo in
      let top_field nstruct fd =
        Structs.replace nstruct fd (invalidate_value ask fd.ftype (Structs.get old fd))
      in
      List.fold_left top_field nstruct compinfo.cfields
    in
    let array_idx_top = (ExpDomain.top (), ArrIdxDomain.top ()) in
    match typ, state with
    |                 _ , `Address n    -> `Address (AD.join AD.top_ptr n)
    | TComp (ci,_)  , `Struct n     -> `Struct (invalid_struct ci n)
    |                 _ , `Struct n     -> `Struct (Structs.map (fun x -> invalidate_value ask voidType x) n)
    | TComp (ci,_)  , `Union (`Lifted fd,n) -> `Union (`Lifted fd, invalidate_value ask fd.ftype n)
    | TArray (t,_,_), `Array n      ->
      let v = invalidate_value ask t (CArrays.get ask n array_idx_top) in
      `Array (CArrays.set ask n (array_idx_top) v)
    |                 _ , `Array n      ->
      let v = invalidate_value ask voidType (CArrays.get ask n (array_idx_top)) in
      `Array (CArrays.set ask n (array_idx_top) v)
    |                 t , `Blob n       -> `Blob (Blobs.invalidate_value ask t n)
    |                 _ , `Thread _     -> state (* TODO: no top thread ID set! *)
    | _, `Bot -> `Bot (* Leave uninitialized value (from malloc) alone in free to avoid trashing everything. TODO: sound? *)
    |                 t , _             -> top_value t


  (* take the last offset in offset and move it over to left *)
  let shift_one_over left offset =
    match left, offset with
    | Some(left), Some(offset) ->
      begin
        (* Remove the first part of an offset, returns (removedPart, remainingOffset) *)
        let removeFirstOffset offset =
          match offset with
            | Field(f, o) -> Field(f, NoOffset), o
            | Index(exp, o) -> Index(exp, NoOffset), o
            | NoOffset -> offset, offset in
        let removed, remaining = removeFirstOffset offset in
        Some (Cil.addOffsetLval removed left), Some(remaining)
      end
    | _ -> None, None

  let determine_offset (ask: Q.ask) left offset exp v =
    let rec contains_pointer exp = (* CIL offsets containing pointers is no issue here, as pointers can only occur in `Index and the domain *)
      match exp with               (* does not partition according to expressions having `Index in them *)
      |	Const _
      |	SizeOf _
      |	SizeOfE _
      |	SizeOfStr _
      |	AlignOf _
      | Lval(Var _, _)
      |	AlignOfE _ -> false
      | Question(e1, e2, e3, _) ->
        (contains_pointer e1) || (contains_pointer e2) || (contains_pointer e3)
      |	CastE(_, e)
      |	UnOp(_, e , _)
      | Real e
      | Imag e -> contains_pointer e
      |	BinOp(_, e1, e2, _) -> (contains_pointer e1) || (contains_pointer e2)
      | AddrOf _
      | AddrOfLabel _
      | StartOf _
      | Lval(Mem _, _) -> true
    in
    let equiv_expr exp start_of_array_lval =
      match exp, start_of_array_lval with
      | BinOp(IndexPI, Lval lval, add, _), (Var arr_start_var, NoOffset) when not (contains_pointer add) ->
        begin
        match ask.f (Q.MayPointTo (Lval lval)) with
        | v when Q.LS.cardinal v = 1 && not (Q.LS.is_top v) ->
          begin
          match Q.LS.choose v with
          | (var,`Index (i,`NoOffset)) when Basetype.CilExp.equal i Cil.zero && CilType.Varinfo.equal var arr_start_var ->
            (* The idea here is that if a must(!) point to arr and we do sth like a[i] we don't want arr to be partitioned according to (arr+i)-&a but according to i instead  *)
            add
          | _ -> BinOp(MinusPP, exp, StartOf start_of_array_lval, !ptrdiffType)
          end
        | _ ->  BinOp(MinusPP, exp, StartOf start_of_array_lval, !ptrdiffType)
        end
      | _ -> BinOp(MinusPP, exp, StartOf start_of_array_lval, !ptrdiffType)
    in
    (* Create a typesig from a type, but drop the arraylen attribute *)
    let typeSigWithoutArraylen t =
      let attrFilter (attr : attribute) : bool =
        match attr with
        | Attr ("arraylen", _) -> false
        | _ -> true
      in
      typeSigWithAttrs (List.filter attrFilter) t
    in
    match left, offset with
      | Some(Var(_), _), Some(Index(exp, _)) -> (* The offset does not matter here, exp is used to index into this array *)
        if not (contains_pointer exp) then
          `Lifted exp
        else
          ExpDomain.top ()
      | Some((Mem(ptr), NoOffset)), Some(NoOffset) ->
        begin
          match v with
          | Some (v') ->
            begin
              try
                (* This should mean the entire expression we have here is a pointer into the array *)
                if Cil.isArrayType (Cilfacade.typeOfLval v') then
                  let expr = ptr in
                  let start_of_array = StartOf v' in
                  let start_type = typeSigWithoutArraylen (Cilfacade.typeOf start_of_array) in
                  let expr_type = typeSigWithoutArraylen (Cilfacade.typeOf ptr) in
                  (* Comparing types for structural equality is incorrect here, use typeSig *)
                  (* as explained at https://people.eecs.berkeley.edu/~necula/cil/api/Cil.html#TYPEtyp *)
                  if start_type = expr_type then
                    `Lifted (equiv_expr expr v')
                  else
                    (* If types do not agree here, this means that we were looking at pointers that *)
                    (* contain more than one array access. Those are not supported. *)
                    ExpDomain.top ()
                else
                  ExpDomain.top ()
              with (Cilfacade.TypeOfError _) -> ExpDomain.top ()
            end
          | _ ->
            ExpDomain.top ()
        end
      | _, _ ->  ExpDomain.top()

  let zero_init_calloced_memory orig x t =
    if orig then
      (* This Blob came from malloc *)
      x
    else if x = `Bot then
      (* This Blob came from calloc *)
      zero_init_value t (* This should be zero initialized *)
    else
      x (* This already contains some value *)

  (* Funny, this does not compile without the final type annotation! *)
  let rec eval_offset (ask: Q.ask) f (x: t) (offs:offs) (exp:exp option) (v:lval option) (t:typ): t =
    let rec do_eval_offset (ask:Q.ask) f (x:t) (offs:offs) (exp:exp option) (l:lval option) (o:offset option) (v:lval option) (t:typ): t =
      match x, offs with
      | `Blob((va, _, orig) as c), `Index (_, ox) ->
        begin
          let l', o' = shift_one_over l o in
          let ev = do_eval_offset ask f (Blobs.value c) ox exp l' o' v t in
          zero_init_calloced_memory orig ev t
        end
      | `Blob((va, _, orig) as c), `Field _ ->
        begin
          let l', o' = shift_one_over l o in
          let ev = do_eval_offset ask f (Blobs.value c) offs exp l' o' v t in
          zero_init_calloced_memory orig ev t
        end
      | `Blob((va, _, orig) as c), `NoOffset ->
      begin
        let l', o' = shift_one_over l o in
        let ev = do_eval_offset ask f (Blobs.value c) offs exp l' o' v t in
        zero_init_calloced_memory orig ev t
      end
      | `Bot, _ -> `Bot
      | _ ->
        match offs with
        | `NoOffset -> x
        | `Field (fld, offs) when fld.fcomp.cstruct -> begin
            match x with
            | `Struct str ->
              let x = Structs.get str fld in
              let l', o' = shift_one_over l o in
              do_eval_offset ask f x offs exp l' o' v t
            | `Top -> M.debug "Trying to read a field, but the struct is unknown"; top ()
            | _ -> M.warn "Trying to read a field, but was not given a struct"; top ()
          end
        | `Field (fld, offs) -> begin
            match x with
            | `Union (`Lifted l_fld, valu) ->
              let x = cast ~torg:l_fld.ftype fld.ftype valu in
              let l', o' = shift_one_over l o in
              do_eval_offset ask f x offs exp l' o' v t
            | `Union (_, valu) -> top ()
            | `Top -> M.debug "Trying to read a field, but the union is unknown"; top ()
            | _ -> M.warn "Trying to read a field, but was not given a union"; top ()
          end
        | `Index (idx, offs) -> begin
            let l', o' = shift_one_over l o in
            match x with
            | `Array x ->
              let e = determine_offset ask l o exp v in
              do_eval_offset ask f (CArrays.get ask x (e, idx)) offs exp l' o' v t
            | `Address _ ->
              begin
                do_eval_offset ask f x offs exp l' o' v t (* this used to be `blob `address -> we ignore the index *)
              end
            | x when GobOption.exists (BI.equal (BI.zero)) (IndexDomain.to_int idx) -> eval_offset ask f x offs exp v t
            | `Top -> M.debug "Trying to read an index, but the array is unknown"; top ()
            | _ -> M.warn "Trying to read an index, but was not given an array (%a)" pretty x; top ()
          end
    in
    let l, o = match exp with
      | Some(Lval (x,o)) -> Some ((x, NoOffset)), Some(o)
      | _ -> None, None
    in
    do_eval_offset ask f x offs exp l o v t

  let update_offset (ask: Q.ask) (x:t) (offs:offs) (value:t) (exp:exp option) (v:lval) (t:typ): t =
    let rec do_update_offset (ask:Q.ask) (x:t) (offs:offs) (value:t) (exp:exp option) (l:lval option) (o:offset option) (v:lval) (t:typ):t =
      if M.tracing then M.traceli "update_offset" "do_update_offset %a %a %a\n" pretty x Offs.pretty offs pretty value;
      let mu = function `Blob (`Blob (y, s', orig), s, orig2) -> `Blob (y, ID.join s s',orig) | x -> x in
      let r =
      match x, offs with
      | `Blob (x,s,orig), `Index (_,ofs) ->
        begin
          let l', o' = shift_one_over l o in
          let x = zero_init_calloced_memory orig x t in
          mu (`Blob (join x (do_update_offset ask x ofs value exp l' o' v t), s, orig))
        end
      | `Blob (x,s,orig), `Field(f, _) ->
        begin
          (* We only have `Blob for dynamically allocated memory. In these cases t is the type of the lval used to access it, i.e. for a struct s {int x; int y;} a; accessed via a->x     *)
          (* will be int. Here, we need a zero_init of the entire contents of the blob though, which we get by taking the associated f.fcomp. Putting [] for attributes is ok, as we don't *)
          (* consider them in VD *)
          let l', o' = shift_one_over l o in
          let x = zero_init_calloced_memory orig x (TComp (f.fcomp, [])) in
          mu (`Blob (join x (do_update_offset ask x offs value exp l' o' v t), s, orig))
        end
      | `Blob (x,s,orig), _ ->
        begin
          let l', o' = shift_one_over l o in
          let x = zero_init_calloced_memory orig x t in
          mu (`Blob (join x (do_update_offset ask x offs value exp l' o' v t), s, orig))
        end
      | `Thread _, _ ->
        (* hack for pthread_t variables *)
        begin match value with
          | `Thread t -> value (* if actually assigning thread, use value *)
          | _ ->
            if !GU.global_initialization then
              `Thread (ConcDomain.ThreadSet.empty ()) (* if assigning global init (int on linux, ptr to struct on mac), use empty set instead *)
            else
              `Top
        end
      | _ ->
      let result =
        match offs with
        | `NoOffset -> begin
            match value with
            | `Blob (y, s, orig) -> mu (`Blob (join x y, s, orig))
            | `Int _ -> cast t value
            | _ -> value
          end
        | `Field (fld, offs) when fld.fcomp.cstruct -> begin
            let t = fld.ftype in
            match x with
            | `Struct str ->
              begin
                let l', o' = shift_one_over l o in
                let value' = do_update_offset ask (Structs.get str fld) offs value exp l' o' v t in
                `Struct (Structs.replace str fld value')
              end
            | `Bot ->
              let init_comp compinfo =
                let nstruct = Structs.create (fun fd -> `Bot) compinfo in
                let init_field nstruct fd = Structs.replace nstruct fd `Bot in
                List.fold_left init_field nstruct compinfo.cfields
              in
              let strc = init_comp fld.fcomp in
              let l', o' = shift_one_over l o in
              `Struct (Structs.replace strc fld (do_update_offset ask `Bot offs value exp l' o' v t))
            | `Top -> M.warn "Trying to update a field, but the struct is unknown"; top ()
            | _ -> M.warn "Trying to update a field, but was not given a struct"; top ()
          end
        | `Field (fld, offs) -> begin
            let t = fld.ftype in
            let l', o' = shift_one_over l o in
            match x with
            | `Union (last_fld, prev_val) ->
              let tempval, tempoffs =
                if UnionDomain.Field.equal last_fld (`Lifted fld) then
                  prev_val, offs
                else begin
                  match offs with
                  | `Field (fldi, _) when fldi.fcomp.cstruct ->
                    (top_value fld.ftype), offs
                  | `Field (fldi, _) -> `Union (Unions.top ()), offs
                  | `NoOffset -> top (), offs
                  | `Index (idx, _) when Cil.isArrayType fld.ftype ->
                    begin
                      match fld.ftype with
                      | TArray(_, l, _) ->
                        let len = try Cil.lenOfArray l
                          with Cil.LenOfArray -> 42 (* will not happen, VLA not allowed in union and struct *) in
                        `Array(CArrays.make (IndexDomain.of_int (Cilfacade.ptrdiff_ikind ()) (BI.of_int len)) `Top), offs
                      | _ -> top (), offs (* will not happen*)
                    end
                  | `Index (idx, _) when IndexDomain.equal idx (IndexDomain.of_int (Cilfacade.ptrdiff_ikind ()) BI.zero) ->
                    (* Why does cil index unions? We'll just pick the first field. *)
                    top (), `Field (List.nth fld.fcomp.cfields 0,`NoOffset)
                  | _ -> M.warn "Why are you indexing on a union? Normal people give a field name.";
                    top (), offs
                end
              in
              `Union (`Lifted fld, do_update_offset ask tempval tempoffs value exp l' o' v t)
            | `Bot -> `Union (`Lifted fld, do_update_offset ask `Bot offs value exp l' o' v t)
            | `Top -> M.warn "Trying to update a field, but the union is unknown"; top ()
            | _ -> M.warn "Trying to update a field, but was not given a union"; top ()
          end
        | `Index (idx, offs) -> begin
            let l', o' = shift_one_over l o in
            match x with
            | `Array x' ->
              let t = (match t with
              | TArray(t1 ,_,_) -> t1
              | _ -> t) in (* This is necessary because t is not a TArray in case of calloc *)
              let e = determine_offset ask l o exp (Some v) in
              let new_value_at_index = do_update_offset ask (CArrays.get ask x' (e,idx)) offs value exp l' o' v t in
              let new_array_value = CArrays.set ask x' (e, idx) new_value_at_index in
              `Array new_array_value
            | `Bot ->
              let t,len = (match t with
                  | TArray(t1 ,len,_) -> t1, len
                  | _ -> t, None) in (* This is necessary because t is not a TArray in case of calloc *)
              let x' = CArrays.bot () in
              let e = determine_offset ask l o exp (Some v) in
              let new_value_at_index = do_update_offset ask `Bot offs value exp l' o' v t in
              let new_array_value =  CArrays.set ask x' (e, idx) new_value_at_index in
              let len_ci = BatOption.bind len (fun e -> Cil.getInteger @@ Cil.constFold true e) in
              let len_id = BatOption.map (fun ci -> IndexDomain.of_int (Cilfacade.ptrdiff_ikind ()) @@ Cilint.big_int_of_cilint ci) len_ci in
              let newl = BatOption.default (ID.starting (Cilfacade.ptrdiff_ikind ()) Z.zero) len_id in
              let new_array_value = CArrays.update_length newl new_array_value in
              `Array new_array_value
            | `Top -> M.warn "Trying to update an index, but the array is unknown"; top ()
            | x when GobOption.exists (BI.equal BI.zero) (IndexDomain.to_int idx) -> do_update_offset ask x offs value exp l' o' v t
            | _ -> M.warn "Trying to update an index, but was not given an array(%a)" pretty x; top ()
          end
      in mu result
      in
      if M.tracing then M.traceu "update_offset" "do_update_offset -> %a\n" pretty r;
      r
    in
    let l, o = match exp with
      | Some(Lval (x,o)) -> Some ((x, NoOffset)), Some(o)
      | _ -> None, None
    in
    do_update_offset ask x offs value exp l o v t

  let rec affect_move ?(replace_with_const=false) ask (x:t) (v:varinfo) movement_for_expr:t =
    let move_fun x = affect_move ~replace_with_const:replace_with_const ask x v movement_for_expr in
    match x with
    | `Array a ->
      begin
        (* potentially move things (i.e. other arrays after arbitrarily deep nesting) in array first *)
        let moved_elems = CArrays.map move_fun a in
        (* then move the array itself *)
        let new_val = CArrays.move_if_affected ~replace_with_const:replace_with_const ask moved_elems v movement_for_expr in
        `Array (new_val)
      end
    | `Struct s -> `Struct (Structs.map (move_fun) s)
    | `Union (f, v) -> `Union(f, move_fun v)
    (* `Blob can not contain Array *)
    | x -> x

  let rec affecting_vars (x:t) =
    let add_affecting_one_level list (va:t) =
      list @ (affecting_vars va)
    in
    match x with
    | `Array a ->
      begin
        let immediately_affecting = CArrays.get_vars_in_e a in
        CArrays.fold_left add_affecting_one_level immediately_affecting a
      end
    | `Struct s ->
        Structs.fold (fun x value acc -> add_affecting_one_level acc value) s []
    | `Union (f, v) ->
        affecting_vars v
    (* `Blob can not contain Array *)
    | _ -> []

  (* Won't compile without the final :t annotation *)
  let rec update_array_lengths (eval_exp: exp -> t) (v:t) (typ:Cil.typ):t =
    match v, typ with
    | `Array(n), TArray(ti, e, _) ->
      begin
        let update_fun x = update_array_lengths eval_exp x ti in
        let n' = CArrays.map (update_fun) n in
        let newl = match e with
          | None -> ID.starting (Cilfacade.ptrdiff_ikind ()) Z.zero
          | Some e ->
            begin
              match eval_exp e with
              | `Int x -> ID.cast_to (Cilfacade.ptrdiff_ikind ())  x
              | _ ->
                M.debug ~category:Analyzer "Expression for size of VLA did not evaluate to Int at declaration";
                ID.starting (Cilfacade.ptrdiff_ikind ()) Z.zero
            end
        in
        `Array(CArrays.update_length newl n')
      end
    | _ -> v


  let printXml f state =
    match state with
    | `Int n ->  ID.printXml f n
    | `Address n ->  AD.printXml f n
    | `Struct n ->  Structs.printXml f n
    | `Union n ->  Unions.printXml f n
    | `Array n ->  CArrays.printXml f n
    | `Blob n ->  Blobs.printXml f n
    | `Thread n -> Threads.printXml f n
    | `Bot -> BatPrintf.fprintf f "<value>\n<data>\nbottom\n</data>\n</value>\n"
    | `Top -> BatPrintf.fprintf f "<value>\n<data>\ntop\n</data>\n</value>\n"

  let to_yojson = function
    | `Int n -> ID.to_yojson n
    | `Address n -> AD.to_yojson n
    | `Struct n -> Structs.to_yojson n
    | `Union n -> Unions.to_yojson n
    | `Array n -> CArrays.to_yojson n
    | `Blob n -> Blobs.to_yojson n
    | `Thread n -> Threads.to_yojson n
    | `Bot -> `String "⊥"
    | `Top -> `String "⊤"

  let invariant c = function
    | `Int n -> ID.invariant c n
    | `Address n -> AD.invariant c n
    | `Blob n -> Blobs.invariant c n
    | `Struct n -> Structs.invariant c n
    | `Union n -> Unions.invariant c n
    | _ -> None (* TODO *)

  let arbitrary () = QCheck.always `Bot (* S TODO: other elements *)

  let rec project p (v: t): t =
    match v with
    | `Int n ->  `Int (ID.project p n)
    | `Address n -> `Address (project_addr p n)
    | `Struct n -> `Struct (Structs.map (fun (x: t) -> project p x) n)
    | `Union (f, v) -> `Union (f, project p v)
    | `Array n -> `Array (project_arr p n)
    | `Blob (v, s, z) -> `Blob (project p v, ID.project p s, z)
    | `Thread n -> `Thread n
    | `Bot -> `Bot
    | `Top -> `Top
  and project_addr p a =
    AD.map (fun addr ->
        match addr with
        | Addr.Addr (v, o) -> Addr.Addr (v, project_offs p o)
        | ptr -> ptr) a
  and project_offs p offs =
    match offs with
    | `NoOffset -> `NoOffset
    | `Field (field, offs') -> `Field (field, project_offs p offs')
    | `Index (idx, offs') -> `Index (ID.project p idx, project_offs p offs')
  and project_arr p n =
    let n' = CArrays.map (fun (x: t) -> project p x) n in
    match CArrays.length n with
    | None -> n'
    | Some l -> CArrays.update_length (ID.project p l) n'
end

and Structs: StructDomain.S with type field = fieldinfo and type value = Compound.t =
  StructDomain.FlagConfiguredStructDomain (Compound)

and Unions: Lattice.S with type t = UnionDomain.Field.t * Compound.t =
  UnionDomain.Simple (Compound)

and CArrays: ArrayDomain.S with type value = Compound.t and type idx = ArrIdxDomain.t =
  ArrayDomain.FlagConfiguredArrayDomain(Compound)(ArrIdxDomain)

and Blobs: Blob with type size = ID.t and type value = Compound.t and type origin = ZeroInit.t = Blob (Compound) (ID)
