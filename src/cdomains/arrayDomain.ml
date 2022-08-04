open Pretty
open Cil
open GobConfig
open FlagHelper

module M = Messages
module A = Array
module Q = Queries
module BI = IntOps.BigIntOps

module LiftExp = Printable.Lift (CilType.Exp) (Printable.DefaultNames)

module type S =
sig
  include Lattice.S
  type idx
  type value

  val get: Q.ask -> t -> ExpDomain.t * idx -> value
  val set: Q.ask -> t -> ExpDomain.t * idx -> value -> t
  val make: idx -> value -> t
  val length: t -> idx option

  val move_if_affected: ?replace_with_const:bool -> Q.ask -> t -> Cil.varinfo -> (Cil.exp -> int option) -> t
  val get_vars_in_e: t -> Cil.varinfo list
  val map: (value -> value) -> t -> t
  val fold_left: ('a -> value -> 'a) -> 'a -> t -> 'a
  val smart_join: (exp -> BI.t option) -> (exp -> BI.t option) -> t -> t -> t
  val smart_widen: (exp -> BI.t option) -> (exp -> BI.t option) -> t -> t -> t
  val smart_leq: (exp -> BI.t option) -> (exp -> BI.t option) -> t -> t -> bool
  val update_length: idx -> t -> t
end

module type LatticeWithSmartOps =
sig
  include Lattice.S
  val smart_join: (Cil.exp -> BI.t option) -> (Cil.exp -> BI.t option) -> t -> t -> t
  val smart_widen: (Cil.exp -> BI.t option) -> (Cil.exp -> BI.t option) -> t -> t -> t
  val smart_leq: (Cil.exp -> BI.t option) -> (Cil.exp -> BI.t option) -> t -> t -> bool
end


module Trivial (Val: Lattice.S) (Idx: Lattice.S): S with type value = Val.t and type idx = Idx.t =
struct
  include Val
  let name () = "trivial arrays"
  type idx = Idx.t
  type value = Val.t

  let show x = "Array: " ^ Val.show x
  let pretty () x = text "Array: " ++ pretty () x
  let pretty_diff () (x,y) = dprintf "%s: %a not leq %a" (name ()) pretty x pretty y
  let get (ask: Q.ask) a i = a
  let set (ask: Q.ask) a i v = join a v
  let make i v = v
  let length _ = None

  let move_if_affected ?(replace_with_const=false) _ x _ _ = x
  let get_vars_in_e _ = []
  let map f x = f x
  let fold_left f a x = f a x

  let printXml f x = BatPrintf.fprintf f "<value>\n<map>\n<key>Any</key>\n%a\n</map>\n</value>\n" Val.printXml x
  let smart_join _ _ = join
  let smart_widen _ _ = widen
  let smart_leq _ _ = leq
  let update_length _ x = x
end

module Unroll (Val: Lattice.S) (Idx:IntDomain.Z): S with type value = Val.t and type idx = Idx.t =
struct
  module Factor = struct let x () = (get_int "ana.base.arrays.unrolling-factor") end
  module Base = Lattice.ProdList (Val) (Factor)
  include Lattice.ProdSimple(Base) (Val)

  let name () = "unrolled arrays"
  type idx = Idx.t
  type value = Val.t
  let factor () =
    match get_int "ana.base.arrays.unrolling-factor" with
    | 0 -> failwith "ArrayDomain: ana.base.arrays.unrolling-factor needs to be set when using the unroll domain"
    | x -> x
  let join_of_all_parts (xl, xr) = List.fold_left Val.join xr xl
  let show (xl, xr) =
    let rec show_list xlist = match xlist with
      | [] -> " --- "
      | hd::tl -> (Val.show hd ^ " - " ^ (show_list tl)) in
    "Array (unrolled to " ^ (Stdlib.string_of_int (factor ())) ^ "): " ^
    (show_list xl) ^ Val.show xr ^ ")"
  let pretty () x = text "Array: " ++ text (show x)
  let pretty_diff () (x,y) = dprintf "%s: %a not leq %a" (name ()) pretty x pretty y
  let extract x = match x with
    | Some c -> c
    | None -> failwith "arrayDomain: that should not happen"
  let get (ask: Q.ask) (xl, xr) (_,i) =
    let search_unrolled_values min_i max_i =
      let mi = Z.to_int min_i in
      let ma = Z.to_int max_i in
      let rec subjoin l i = match l with
        | [] -> Val.bot ()
        | hd::tl ->
          begin
            match i>ma,i<mi with
            | false,true -> subjoin tl (i+1)
            | false,false -> Val.join hd (subjoin tl (i+1))
            | _,_ -> Val.bot ()
          end in
      subjoin xl 0 in
    let f = Z.of_int (factor ()) in
    let min_i = extract (Idx.minimal i) in
    let max_i = extract (Idx.maximal i) in
    if Z.geq min_i f then xr
    else if Z.lt max_i f then search_unrolled_values min_i max_i
    else Val.join xr (search_unrolled_values min_i (Z.of_int ((factor ())-1)))
  let set (ask: Q.ask) (xl,xr) (_,i) v =
    let update_unrolled_values min_i max_i =
      let mi = Z.to_int min_i in
      let ma = Z.to_int max_i in
      let rec weak_update l i = match l with
        | [] -> []
        | hd::tl ->
          if i<mi then hd::(weak_update tl (i+1))
          else if i>ma then (hd::tl)
          else (Val.join hd v)::(weak_update tl (i+1)) in
      let rec full_update l i = match l with
        | [] -> []
        | hd::tl ->
          if i<mi then hd::(full_update tl (i+1))
          else v::tl in
      if mi=ma then full_update xl 0
      else weak_update xl 0 in
    let f = Z.of_int (factor ()) in
    let min_i = extract(Idx.minimal i) in
    let max_i = extract(Idx.maximal i) in
    if Z.geq min_i f then (xl, (Val.join xr v))
    else if Z.lt max_i f then ((update_unrolled_values min_i max_i), xr)
    else ((update_unrolled_values min_i (Z.of_int ((factor ())-1))), (Val.join xr v))
  let make _ v =
    let xl = BatList.make (factor ()) v in
    (xl,Val.bot ())
  let length _ = None
  let move_if_affected ?(replace_with_const=false) _ x _ _ = x
  let get_vars_in_e _ = []
  let map f (xl, xr) = ((List.map f xl), f xr)
  let fold_left f a x = f a (join_of_all_parts x)
  let printXml f (xl,xr) = BatPrintf.fprintf f "<value>\n<map>\n
  <key>unrolled array</key>\n
  <key>xl</key>\n%a\n\n
  <key>xm</key>\n%a\n\n
  </map></value>\n" Base.printXml xl Val.printXml xr
  let smart_join _ _ = join
  let smart_widen _ _ = widen
  let smart_leq _ _ = leq
  let update_length _ x = x
end

(** Special signature so that we can use the _with_length functions from PartitionedWithLength but still match the interface *
  * defined for array domains *)
module type SPartitioned =
sig
  include S
  val set_with_length: idx option -> Q.ask -> t -> ExpDomain.t * idx -> value -> t
  val smart_join_with_length: idx option -> (exp -> BI.t option) -> (exp -> BI.t option) -> t -> t -> t
  val smart_widen_with_length: idx option -> (exp -> BI.t option) -> (exp -> BI.t option)  -> t -> t-> t
  val smart_leq_with_length: idx option -> (exp -> BI.t option) -> (exp -> BI.t option) -> t -> t -> bool
  val move_if_affected_with_length: ?replace_with_const:bool -> idx option -> Q.ask -> t -> Cil.varinfo -> (Cil.exp -> int option) -> t
end

module Partitioned (Val: LatticeWithSmartOps) (Idx:IntDomain.Z):SPartitioned with type value = Val.t and type idx = Idx.t =
struct
  (* Contrary to the description in Michael's master thesis, abstract values here always have the form *)
  (* (Expp, (Val, Val, Val)). Expp is top when the array is not partitioned. In these cases all three  *)
  (* values from Val are identical *)
  module Expp = ExpDomain
  module Base = Lattice.Prod3 (Val) (Val) (Val)
  include Lattice.ProdSimple(Expp) (Base)

  type idx = Idx.t
  type value = Val.t

  let name () = "partitioned array"

  let is_not_partitioned (e, _) =
    Expp.is_bot e || Expp.is_top e

  let join_of_all_parts (_,(xl, xm, xr)) =
    let result = Val.join (Val.join xl xm) xr
    in
    if Val.is_bot result then
      Val.top()
    else
      result

  (** Ensures an array where all three Val are equal, is represented by an unpartitioned array and all unpartitioned arrays
    * have the same three values for Val  *)
  let normalize ((e, (xl, xm , xr)) as x) =
    if Val.equal xl xm && Val.equal xm xr then
      (Expp.top (), (xl, xm, xr))
    else if Expp.is_top e then
      (Expp.top(), (join_of_all_parts x, join_of_all_parts x, join_of_all_parts x))
    else
      x

  let show ((e,(xl, xm, xr)) as x) =
    if is_not_partitioned x then
      "Array (no part.): " ^ Val.show xl
    else
      "Array (part. by " ^ Expp.show e ^ "): (" ^
        Val.show xl ^ " -- " ^
        Val.show xm ^ " -- " ^
        Val.show xr ^ ")"

  let pretty () x = text "Array: " ++ text (show x)
  let pretty_diff () (x,y) = dprintf "%s: %a not leq %a" (name ()) pretty x pretty y

  let printXml f ((e, (xl, xm, xr)) as x) =
    if is_not_partitioned x then
      let join_over_all = Val.join (Val.join xl xm) xr in
      BatPrintf.fprintf f "<value>\n<map>\n<key>Any</key>\n%a\n</map>\n</value>\n" Val.printXml join_over_all
    else
      BatPrintf.fprintf f "<value>\n<map>\n
          <key>Partitioned By</key>\n%a\n
          <key>l</key>\n%a\n\n
          <key>m</key>\n%a\n\n
          <key>r</key>\n%a\n\n
        </map></value>\n" Expp.printXml e Val.printXml xl Val.printXml xm Val.printXml xr

  let to_yojson ((e, (l, m, r)) as x) =
    if is_not_partitioned x then
      let join_over_all = Val.join (Val.join l m) r in
      `Assoc [ ("any", Val.to_yojson join_over_all) ]
    else
      let e' = Expp.to_yojson e in
      let l' = Val.to_yojson l in
      let m' = Val.to_yojson m in
      let r' = Val.to_yojson r in
      `Assoc [ ("partitioned by", e'); ("l", l'); ("m", m'); ("r", r') ]

  let get (ask:Q.ask) ((e, (xl, xm, xr)) as x) (i,_) =
    match e, i with
    | `Lifted e', `Lifted i' ->
      begin
        if ask.f (Q.MustBeEqual (e',i')) then xm
        else
          begin
            let contributionLess = match ask.f (Q.MayBeLess (i', e')) with        (* (may i < e) ? xl : bot *)
            | false -> Val.bot ()
            | _ -> xl in
            let contributionEqual = match ask.f (Q.MayBeEqual (i', e')) with      (* (may i = e) ? xm : bot *)
            | false -> Val.bot ()
            | _ -> xm in
            let contributionGreater =  match ask.f (Q.MayBeLess (e', i')) with    (* (may i > e) ? xr : bot *)
            | false -> Val.bot ()
            | _ -> xr in
            Val.join (Val.join contributionLess contributionEqual) contributionGreater
          end
      end
    | _ -> join_of_all_parts x

  let get_vars_in_e (e, _) =
    match e with
    | `Top
    | `Bot -> []
    | `Lifted exp -> Basetype.CilExp.get_vars exp

  (* expressions containing globals or array accesses are not suitable for partitioning *)
  let not_allowed_for_part e =
    let rec contains_array_access e =
      let rec offset_contains_array_access offs =
        match offs with
        | NoOffset -> false
        | Index _ -> true
        | Field (_, o) -> offset_contains_array_access o
      in
      match e with
        |	Const _
        |	SizeOf _
        |	SizeOfE _
        |	SizeOfStr _
        |	AlignOf _
        |	AlignOfE _ -> false
        | Question(e1, e2, e3, _) ->
          contains_array_access e1 || contains_array_access e2 || contains_array_access e3
        |	CastE(_, e)
        |	UnOp(_, e , _)
        | Real e
        | Imag e -> contains_array_access e
        |	BinOp(_, e1, e2, _) -> contains_array_access e1 || contains_array_access e2
        | AddrOf _
        | AddrOfLabel _
        | StartOf _ -> false
        | Lval(Mem e, o) -> offset_contains_array_access o || contains_array_access e
        | Lval(Var _, o) -> offset_contains_array_access o
    in
    match e with
    | `Top
    | `Bot -> true
    | `Lifted exp ->
      let vars = Basetype.CilExp.get_vars exp in
      List.exists (fun x -> x.vglob) vars || contains_array_access exp


  let map f (e, (xl, xm, xr)) =
    normalize @@ (e, (f xl, f xm, f xr))

  let fold_left f a (_, ((xl:value), (xm:value), (xr:value))) =
    f (f (f a xl) xm) xr

  let move_if_affected_with_length ?(replace_with_const=false) length (ask:Q.ask) ((e, (xl,xm, xr)) as x) (v:varinfo) movement_for_exp =
    normalize @@
    let move (i:int option) =
      match i with
      | Some 0   ->
        (e, (xl, xm, xr))
      | Some 1   ->
        (e, (Val.join xl xm, xr, xr)) (* moved one to the right *)
      | Some -1  ->
        (e, (xl, xl, Val.join xm xr)) (* moved one to the left  *)
      | Some x when x > 1 ->
        (e, (Val.join (Val.join xl xm) xr, xr, xr)) (* moved more than one to the right *)
      | Some x when x < -1 ->
        (e, (xl, xl, Val.join (Val.join xl xm) xr)) (* moved more than one to the left *)
      | _ ->
        begin
          let nval = join_of_all_parts x in
          let default = (Expp.top (), (nval, nval, nval)) in
          if replace_with_const then
            match e with
            | `Lifted e' ->
              begin
                let n = ask.f (Q.EvalInt e') in
                match Q.ID.to_int n with
                | Some i ->
                  (`Lifted (Cil.kintegerCilint (Cilfacade.ptrdiff_ikind ()) i), (xl, xm, xr))
                | _ -> default
              end
            | _ -> default
          else
            default
        end
    in
    match e with
    | `Lifted exp ->
        let is_affected = Basetype.CilExp.occurs v exp in
        if not is_affected then
          x
        else
          (* check if one part covers the entire array, so we can drop partitioning *)
          begin
            let e_must_bigger_max_index =
              match length with
              | Some l ->
                begin
                  match Idx.to_int l with
                  | Some i ->
                    let b = ask.f (Q.MayBeLess (exp, Cil.kintegerCilint (Cilfacade.ptrdiff_ikind ()) i)) in
                    not b (* !(e <_{may} length) => e >=_{must} length *)
                  | None -> false
                end
              | _ -> false
            in
            let e_must_less_zero =
              let b = ask.f (Q.MayBeLess (Cil.mone, exp)) in
              not b (* !(-1 <_{may} e) => e <=_{must} -1 *)
            in
            if e_must_bigger_max_index then
              (* Entire array is covered by left part, dropping partitioning. *)
              Expp.top(),(xl, xl, xl)
            else if e_must_less_zero then
              (* Entire array is covered by right value, dropping partitioning. *)
              Expp.top(),(xr, xr, xr)
            else
              (* If we can not drop partitioning, move *)
              move (movement_for_exp exp)
          end
    | _ -> x (* If the array is not partitioned, nothing to do *)

  let move_if_affected ?replace_with_const = move_if_affected_with_length ?replace_with_const None

  let set_with_length length (ask:Q.ask) ((e, (xl, xm, xr)) as x) (i,_) a =
    if M.tracing then M.trace "update_offset" "part array set_with_length %a %a %a\n" pretty x LiftExp.pretty i Val.pretty a;
    if i = `Lifted MyCFG.all_array_index_exp then
      (assert !Goblintutil.global_initialization; (* just joining with xm here assumes that all values will be set, which is guaranteed during inits *)
       (* the join is needed here! see e.g 30/04 *)
      let r =  Val.join xm a in
      (Expp.top(), (r, r, r)))
    else
      normalize @@
      let use_last = get_string "ana.base.partition-arrays.keep-expr" = "last" in
      let exp_value e =
        match e with
        | `Lifted e' ->
          let n = ask.f (Q.EvalInt e') in
          Option.map BI.of_bigint (Q.ID.to_int n)
        |_ -> None
      in
      let equals_zero e = BatOption.map_default (BI.equal BI.zero) false (exp_value e) in
      let equals_maxIndex e =
        match length with
        | Some l ->
          begin
            match Idx.to_int l with
            | Some i -> BatOption.map_default (BI.equal (BI.sub i BI.one)) false (exp_value e)
            | None -> false
          end
        | _ -> false
      in
      let lubIfNotBot x = if Val.is_bot x then x else Val.join a x in
      if is_not_partitioned x then
        if not_allowed_for_part i then
          let result = Val.join a (join_of_all_parts x) in
          (e, (result, result, result))
        else
          let l = if equals_zero i then Val.bot () else join_of_all_parts x in
          let r = if equals_maxIndex i then Val.bot () else join_of_all_parts x in
          (i, (l, a, r))
      else
        let isEqual e' i' = ask.f (Q.MustBeEqual (e',i')) in
        match e, i with
        | `Lifted e', `Lifted i' when not use_last || not_allowed_for_part i -> begin
            let default =
              let left =
                match ask.f (Q.MayBeLess (i', e')) with     (* (may i < e) ? xl : bot *)
                | false -> xl
                | _ -> lubIfNotBot xl in
              let middle =
                match ask.f (Q.MayBeEqual (i', e')) with    (* (may i = e) ? xm : bot *)
                | false -> xm
                | _ -> Val.join xm a in
              let right =
                match ask.f (Q.MayBeLess (e', i')) with     (* (may i > e) ? xr : bot *)
                | false -> xr
                | _ -> lubIfNotBot xr in
              (e, (left, middle, right))
            in
            if isEqual e' i' then
              (*  e = _{must} i => update strongly *)
              (e, (xl, a, xr))
            else if Cil.isConstant e' && Cil.isConstant i' then
              match Cil.getInteger e', Cil.getInteger i' with
                | Some (e'': Cilint.cilint), Some i'' ->
                  let (i'': BI.t) = Cilint.big_int_of_cilint  i'' in
                  let (e'': BI.t) = Cilint.big_int_of_cilint  e'' in

                  if BI.equal  i'' (BI.add e'' BI.one) then
                    (* If both are integer constants and they are directly adjacent, we change partitioning to maintain information *)
                    (i, (Val.join xl xm, a, xr))
                  else if BI.equal e'' (BI.add i'' BI.one) then
                    (i, (xl, a, Val.join xm xr))
                  else
                    default
                | _ ->
                  default
            else
              default
          end
        | `Lifted e', `Lifted i' ->
          if isEqual e' i' then
            (e,(xl,a,xr))
          else
            let left = if equals_zero i then Val.bot () else Val.join xl @@ Val.join
              (match ask.f (Q.MayBeEqual (e', i')) with
              | false -> Val.bot()
              | _ -> xm) (* if e' may be equal to i', but e' may not be smaller than i' then we only need xm *)
              (
                let t = Cilfacade.typeOf e' in
                let ik = Cilfacade.get_ikind t in
                match ask.f (Q.MustBeEqual(BinOp(PlusA, e', Cil.kinteger ik 1, t),i')) with
                | true -> xm
                | _ ->
                  begin
                    match ask.f (Q.MayBeLess (e', i')) with
                    | false-> Val.bot()
                    | _ -> Val.join xm xr (* if e' may be less than i' then we also need xm for sure *)
                  end
              )
            in
            let right = if equals_maxIndex i then Val.bot () else  Val.join xr @@  Val.join
              (match ask.f (Q.MayBeEqual (e', i')) with
              | false -> Val.bot()
              | _ -> xm)

              (
                let t = Cilfacade.typeOf e' in
                let ik = Cilfacade.get_ikind t in
                match ask.f (Q.MustBeEqual(BinOp(PlusA, e', Cil.kinteger ik (-1), t),i')) with
                | true -> xm
                | _ ->
                  begin
                    match ask.f (Q.MayBeLess (i', e')) with
                    | false -> Val.bot()
                    | _ -> Val.join xl xm (* if e' may be less than i' then we also need xm for sure *)
                  end
              )
            in
            (* The new thing is partitioned according to i so we can strongly update *)
            (i,(left, a, right))
        | _ ->
          (* If the expression used to write is not known, all segments except the empty ones will be affected *)
          (e, (lubIfNotBot xl, Val.join xm a, lubIfNotBot xr))

  let set = set_with_length None

  let join ((e1, (xl1,xm1,xr1)) as x1) ((e2, (xl2,xm2,xr2)) as x2) =
    normalize @@ let new_e = Expp.join e1 e2 in
    if Expp.is_top new_e then
      (* At least one of them was not partitioned, or e <> f *)
      let join_over_all = Val.join (join_of_all_parts x1) (join_of_all_parts x2) in
      (new_e, (join_over_all, join_over_all, join_over_all))
    else
      (new_e, (Val.join xl1 xl2, Val.join xm1 xm2, Val.join xr1 xr2))

  (* leq needs not be given explicitly, leq from product domain works here *)

  let make i v =
    if Idx.to_int i = Some BI.one  then
      (`Lifted (Cil.integer 0), (v, v, v))
    else if Val.is_bot v then
      (Expp.top(), (Val.bot(), Val.bot(), Val.bot()))
    else
      (Expp.top(), (v, v, v))

  let length _ = None

  let smart_op (op: Val.t -> Val.t -> Val.t) length ((e1, (xl1,xm1,xr1)) as x1) ((e2, (xl2,xm2,xr2)) as x2) x1_eval_int x2_eval_int =
    normalize @@
    let must_be_length_minus_one v = match length with
      | Some l ->
        begin
          match Idx.to_int l with
          | Some i ->
            v = Some (BI.sub i BI.one)
          | None -> false
        end
      | None -> false
    in
    let must_be_zero v = v = Some BI.zero in
    let op_over_all = op (join_of_all_parts x1) (join_of_all_parts x2) in
    match e1, e2 with
    | `Lifted e1e, `Lifted e2e when Basetype.CilExp.equal e1e e2e ->
      (e1, (op xl1 xl2, op xm1 xm2, op xr1 xr2))
    | `Lifted e1e, `Lifted e2e ->
      if get_string "ana.base.partition-arrays.keep-expr" = "last" || get_bool "ana.base.partition-arrays.smart-join" then
        let op = Val.join in (* widen between different components isn't called validly *)
        let over_all_x1 = op (op xl1 xm1) xr1 in
        let over_all_x2 = op (op xl2 xm2) xr2 in
        let e1e_in_state_of_x2 = x2_eval_int e1e in
        let e2e_in_state_of_x1 = x1_eval_int e2e in
        let e1e_is_better = (not (Cil.isConstant e1e) && Cil.isConstant e2e) || Basetype.CilExp.compare e1e e2e < 0 in (* TODO: why does this depend on exp comparison? probably to use "simpler" expression according to constructor order in compare *)
        if e1e_is_better then (* first try if the result can be partitioned by e1e *)
          if must_be_zero e1e_in_state_of_x2  then
            (e1, (xl1, op xm1 over_all_x2, op xr1 over_all_x2))
          else if must_be_length_minus_one e1e_in_state_of_x2  then
            (e1, (op xl1 over_all_x2, op xm1 over_all_x2, xr1))
          else if must_be_zero e2e_in_state_of_x1 then
            (e2, (xl2, op over_all_x1 xm2, op over_all_x1 xr2))
          else if must_be_length_minus_one e2e_in_state_of_x1 then
            (e2, (op over_all_x1 xl2, op over_all_x1 xm2, xr2))
          else
            (Expp.top (), (op_over_all, op_over_all, op_over_all))
        else  (* first try if the result can be partitioned by e2e *)
          if must_be_zero e2e_in_state_of_x1 then
            (e2, (xl2, op over_all_x1 xm2, op over_all_x1 xr2))
          else if must_be_length_minus_one e2e_in_state_of_x1 then
            (e2, (op over_all_x1 xl2, op over_all_x1 xm2, xr2))
          else if must_be_zero e1e_in_state_of_x2 then
            (e1, (xl1, op xm1 over_all_x2, op xr1 over_all_x2))
          else if must_be_length_minus_one e1e_in_state_of_x2 then
            (e1, (op xl1 over_all_x2, op xm1 over_all_x2, xr1))
          else
            (Expp.top (), (op_over_all, op_over_all, op_over_all))
      else
        (Expp.top (), (op_over_all, op_over_all, op_over_all))
    | `Top, `Top ->
      (Expp.top (), (op_over_all, op_over_all, op_over_all))
    | `Top, `Lifted e2e ->
      if must_be_zero (x1_eval_int e2e) then
        (e2, (xl2, op xm1 xm2, op xr1 xr2))
      else if must_be_length_minus_one (x1_eval_int e2e) then
        (e2, (op xl1 xl2, op xm1 xm2, xr2))
      else
        (Expp.top (), (op_over_all, op_over_all, op_over_all))
    | `Lifted e1e, `Top ->
      if must_be_zero (x2_eval_int e1e) then
        (e1, (xl1, op xm1 xm2, op xr1 xr2))
      else if must_be_length_minus_one (x2_eval_int e1e) then
        (e1, (op xl1 xl2, op xm1 xm2, xr1))
      else
        (Expp.top (), (op_over_all, op_over_all, op_over_all))
    | _ ->
      failwith "ArrayDomain: Unallowed state (one of the partitioning expressions is bot)"

  let smart_join_with_length length x1_eval_int x2_eval_int x1 x2 =
    smart_op (Val.smart_join x1_eval_int x2_eval_int) length x1 x2 x1_eval_int x2_eval_int

  let smart_widen_with_length length x1_eval_int x2_eval_int x1 x2  =
    smart_op (Val.smart_widen x1_eval_int x2_eval_int) length x1 x2 x1_eval_int x2_eval_int

  let smart_leq_with_length length x1_eval_int x2_eval_int ((e1, (xl1,xm1,xr1)) as x1) (e2, (xl2, xm2, xr2)) =
    let leq' = Val.smart_leq x1_eval_int x2_eval_int in
    let must_be_zero v = (v = Some BI.zero) in
    let must_be_length_minus_one v =  match length with
      | Some l ->
        begin
          match Idx.to_int l with
          | Some i ->
            v = Some (BI.sub i BI.one)
          | None -> false
        end
      | None -> false
    in
    match e1, e2 with
    | `Top, `Top ->
      (* Those asserts ensure that for both arguments all segments are equal (as it should be) *)
      assert(Val.equal xl1 xm1); assert(Val.equal xm1 xr1); assert(Val.equal xl2 xm2); assert(Val.equal xm2 xr2);
      leq' (Val.join xl1 (Val.join xm1 xr1)) (Val.join xl2 (Val.join xm2 xr2))    (* TODO: should the inner joins also be smart joins? *)
    | `Lifted _, `Top -> leq' (Val.join xl1 (Val.join xm1 xr1)) (Val.join xl2 (Val.join xm2 xr2))
    | `Lifted e1e, `Lifted e2e ->
      if Basetype.CilExp.equal e1e e2e then
        leq' xl1 xl2 && leq' xm1 xm2 && leq' xr1 xr2
      else if must_be_zero (x1_eval_int e2e) then
        (* A read will never be from xl2 -> we can ignore that here *)
        let l = join_of_all_parts x1 in
        leq' l xm2 && leq' l xr2
      else if must_be_length_minus_one (x1_eval_int e2e) then
        (* A read will never be from xr2 -> we can ignore that here *)
        let l = join_of_all_parts x1 in
        leq' l xl2 && leq' l xm2
      else
        false
    | `Top, `Lifted e2e ->
      if must_be_zero (x1_eval_int e2e) then
        leq' xm1 xm2 && leq' xr1 xr2
      else if must_be_length_minus_one (x1_eval_int e2e) then
        leq' xl1 xl2 && leq' xm1 xm2
      else
        false
    | _ ->
      failwith "ArrayDomain: Unallowed state (one of the partitioning expressions is bot)"

  let smart_join = smart_join_with_length None
  let smart_widen = smart_widen_with_length None
  let smart_leq = smart_leq_with_length None

  let meet (e1,v1) (e2,v2) = normalize @@
    match e1,e2 with
    | `Lifted e1e, `Lifted e2e when not (Basetype.CilExp.equal e1e e2e) ->
      (* partitioned according to two different expressions -> meet can not be element-wise *)
      (* arrays can not be partitioned according to multiple expressions, arbitrary prefer the first one here *)
      (* TODO: do smart things if the relationship between e1e and e2e is known *)
      (e1,v1)
    | _ -> meet (e1,v1) (e2,v2)

  let narrow (e1,v1) (e2,v2) = normalize @@
    match e1,e2 with
    | `Lifted e1e, `Lifted e2e when not (Basetype.CilExp.equal e1e e2e) ->
      (* partitioned according to two different expressions -> narrow can not be element-wise *)
      (* arrays can not be partitioned according to multiple expressions, arbitrary prefer the first one here *)
      (* TODO: do smart things if the relationship between e1e and e2e is known *)
      (e1,v1)
    | _ -> narrow (e1,v1) (e2,v2)

  let update_length _ x = x
end
(* This is the main array out of bounds check *)
let array_oob_check ( type a ) (module Idx: IntDomain.Z with type t = a) (x, l) (e, v) =
  if GobConfig.get_bool "ana.arrayoob" then (* The purpose of the following 2 lines is to give the user extra info about the array oob *)
    let idx_before_end = Idx.to_bool (Idx.lt v l) (* check whether index is before the end of the array *)
    and idx_after_start = Idx.to_bool (Idx.ge v (Idx.of_int Cil.ILong BI.zero)) in (* check whether the index is non-negative *)
    (* For an explanation of the warning types check the Pull Request #255 *)
    match(idx_after_start, idx_before_end) with
    | Some true, Some true -> (* Certainly in bounds on both sides.*)
      ()
    | Some true, Some false -> (* The following matching differentiates the must and may cases*)
      M.error ~category:M.Category.Behavior.Undefined.ArrayOutOfBounds.past_end "Must access array past end"
    | Some true, None ->
      M.warn ~category:M.Category.Behavior.Undefined.ArrayOutOfBounds.past_end "May access array past end"
    | Some false, Some true ->
      M.error ~category:M.Category.Behavior.Undefined.ArrayOutOfBounds.before_start "Must access array before start"
    | None, Some true ->
      M.warn ~category:M.Category.Behavior.Undefined.ArrayOutOfBounds.before_start "May access array before start"
    | _ ->
      M.warn ~category:M.Category.Behavior.Undefined.ArrayOutOfBounds.unknown "May access array out of bounds"
  else ()


module TrivialWithLength (Val: Lattice.S) (Idx: IntDomain.Z): S with type value = Val.t and type idx = Idx.t =
struct
  module Base = Trivial (Val) (Idx)
  include Lattice.Prod (Base) (Idx)
  type idx = Idx.t
  type value = Val.t

  let get (ask : Q.ask) (x, (l : idx)) ((e: ExpDomain.t), v) =
    (array_oob_check (module Idx) (x, l) (e, v));
    Base.get ask x (e, v)
  let set (ask: Q.ask) (x,l) i v = Base.set ask x i v, l
  let make l x = Base.make l x, l
  let length (_,l) = Some l

  let move_if_affected ?(replace_with_const=false) _ x _ _ = x
  let map f (x, l):t = (Base.map f x, l)
  let fold_left f a (x, l) = Base.fold_left f a x
  let get_vars_in_e _ = []

  let smart_join _ _ = join
  let smart_widen _ _ = widen
  let smart_leq _ _ = leq

  (* It is not necessary to do a least-upper bound between the old and the new length here.   *)
  (* Any array can only be declared in one location. The value for newl that we get there is  *)
  (* the one obtained by abstractly evaluating the size expression at this location for the   *)
  (* current state. If newl leq l this means that we somehow know more about the expression   *)
  (* determining the size now (e.g. because of narrowing), but this holds for all the times   *)
  (* the declaration is visited. *)
  let update_length newl (x, l) = (x, newl)

  let printXml f (x,y) =
    BatPrintf.fprintf f "<value>\n<map>\n<key>\n%s\n</key>\n%a<key>\n%s\n</key>\n%a</map>\n</value>\n" (XmlUtil.escape (Base.name ())) Base.printXml x "length" Idx.printXml y

  let to_yojson (x, y) = `Assoc [ (Base.name (), Base.to_yojson x); ("length", Idx.to_yojson y) ]
end


module PartitionedWithLength (Val: LatticeWithSmartOps) (Idx: IntDomain.Z): S with type value = Val.t and type idx = Idx.t =
struct
  module Base = Partitioned (Val) (Idx)
  include Lattice.Prod (Base) (Idx)
  type idx = Idx.t
  type value = Val.t

  let get (ask : Q.ask) (x, (l : idx)) ((e: ExpDomain.t), v) =
    (array_oob_check (module Idx) (x, l) (e, v));
    Base.get ask x (e, v)
  let set ask (x,l) i v = Base.set_with_length (Some l) ask x i v, l
  let make l x = Base.make l x, l
  let length (_,l) = Some l

  let move_if_affected ?replace_with_const ask (x,l) v i =
    (Base.move_if_affected_with_length ?replace_with_const (Some l) ask x v i), l

  let map f (x, l):t = (Base.map f x, l)
  let fold_left f a (x, l) = Base.fold_left f a x
  let get_vars_in_e (x, _) = Base.get_vars_in_e x

  let smart_join x_eval_int y_eval_int (x,xl) (y,yl) =
    let l = Idx.join xl yl in
    (Base.smart_join_with_length (Some l) x_eval_int y_eval_int x y , l)

  let smart_widen x_eval_int y_eval_int (x,xl) (y,yl) =
    let l = Idx.join xl yl in
    (Base.smart_widen_with_length (Some l) x_eval_int y_eval_int x y, l)

  let smart_leq x_eval_int y_eval_int (x,xl) (y,yl)  =
    let l = Idx.join xl yl in
    Idx.leq xl yl && Base.smart_leq_with_length (Some l) x_eval_int y_eval_int x y

  (* It is not necessary to do a least-upper bound between the old and the new length here.   *)
  (* Any array can only be declared in one location. The value for newl that we get there is  *)
  (* the one obtained by abstractly evaluating the size expression at this location for the   *)
  (* current state. If newl leq l this means that we somehow know more about the expression   *)
  (* determining the size now (e.g. because of narrowing), but this holds for all the times   *)
  (* the declaration is visited. *)
  let update_length newl (x, l) = (x, newl)

  let printXml f (x,y) =
    BatPrintf.fprintf f "<value>\n<map>\n<key>\n%s\n</key>\n%a<key>\n%s\n</key>\n%a</map>\n</value>\n" (XmlUtil.escape (Base.name ())) Base.printXml x "length" Idx.printXml y

  let to_yojson (x, y) = `Assoc [ (Base.name (), Base.to_yojson x); ("length", Idx.to_yojson y) ]
end

module UnrollWithLength (Val: Lattice.S) (Idx: IntDomain.Z): S with type value = Val.t and type idx = Idx.t =
struct
  module Base = Unroll (Val) (Idx)
  include Lattice.Prod (Base) (Idx)
  type idx = Idx.t
  type value = Val.t

  let get (ask : Q.ask) (x, (l : idx)) ((e: ExpDomain.t), v) =
    (array_oob_check (module Idx) (x, l) (e, v));
    Base.get ask x (e, v)
  let set (ask: Q.ask) (x,l) i v = Base.set ask x i v, l
  let make l x = Base.make l x, l
  let length (_,l) = Some l

  let move_if_affected ?(replace_with_const=false) _ x _ _ = x
  let map f (x, l):t = (Base.map f x, l)
  let fold_left f a (x, l) = Base.fold_left f a x
  let get_vars_in_e _ = []

  let smart_join _ _ = join
  let smart_widen _ _ = widen
  let smart_leq _ _ = leq

  (* It is not necessary to do a least-upper bound between the old and the new length here.   *)
  (* Any array can only be declared in one location. The value for newl that we get there is  *)
  (* the one obtained by abstractly evaluating the size expression at this location for the   *)
  (* current state. If newl leq l this means that we somehow know more about the expression   *)
  (* determining the size now (e.g. because of narrowing), but this holds for all the times   *)
  (* the declaration is visited. *)
  let update_length newl (x, l) = (x, newl)

  let printXml f (x,y) =
    BatPrintf.fprintf f "<value>\n<map>\n<key>\n%s\n</key>\n%a<key>\n%s\n</key>\n%a</map>\n</value>\n" (XmlUtil.escape (Base.name ())) Base.printXml x "length" Idx.printXml y

  let to_yojson (x, y) = `Assoc [ (Base.name (), Base.to_yojson x); ("length", Idx.to_yojson y) ]
end

module FlagConfiguredArrayDomain(Val: LatticeWithSmartOps) (Idx:IntDomain.Z):S with type value = Val.t and type idx = Idx.t =
struct
  module P = PartitionedWithLength(Val)(Idx)
  module T = TrivialWithLength(Val)(Idx)
  module U = UnrollWithLength(Val)(Idx)

  type idx = Idx.t
  type value = Val.t

  module K = struct
    let msg = "FlagConfiguredArrayDomain received a value where not exactly one component is set"
    let name = "FlagConfiguredArrayDomain"
  end

  let to_t = function
    | (Some p, None, None) -> (Some p, None)
    | (None, Some t, None) -> (None, Some (Some t, None))
    | (None, None, Some u) -> (None, Some (None, Some u))
    | _ -> failwith "FlagConfiguredArrayDomain received a value where not exactly one component is set"

  module I = struct include LatticeFlagHelper (T) (U) (K) let name () = "" end
  include LatticeFlagHelper (P) (I) (K)

  let binop' opp opt opu = binop opp (I.binop opt opu)
  let unop' opp opt opu = unop opp (I.unop opt opu)
  let binop_to_t' opp opt opu = binop_to_t opp (I.binop_to_t opt opu)
  let unop_to_t' opp opt opu = unop_to_t opp (I.unop_to_t opt opu)

  (* Simply call appropriate function for component that is not None *)
  let get a x (e,i) = unop' (fun x ->
      if e = `Top then
        let e' = BatOption.map_default (fun x -> `Lifted (Cil.kintegerCilint (Cilfacade.ptrdiff_ikind ()) x)) (`Top) (Idx.to_int i) in
        P.get a x (e', i)
      else
        P.get a x (e, i)
    ) (fun x -> T.get a x (e,i)) (fun x -> U.get a x (e,i)) x
  let set (ask:Q.ask) x i a = unop_to_t' (fun x -> P.set ask x i a) (fun x -> T.set ask x i a) (fun x -> U.set ask x i a) x
  let length = unop' P.length T.length U.length
  let map f = unop_to_t' (P.map f) (T.map f) (U.map f)
  let fold_left f s = unop' (P.fold_left f s) (T.fold_left f s) (U.fold_left f s)

  let move_if_affected ?(replace_with_const=false) (ask:Q.ask) x v f = unop_to_t' (fun x -> P.move_if_affected ~replace_with_const:replace_with_const ask x v f) (fun x -> T.move_if_affected ~replace_with_const:replace_with_const ask x v f) (fun x -> U.move_if_affected ~replace_with_const:replace_with_const ask x v f) x
  let get_vars_in_e = unop' P.get_vars_in_e T.get_vars_in_e U.get_vars_in_e
  let smart_join f g = binop_to_t' (P.smart_join f g) (T.smart_join f g) (U.smart_join f g)
  let smart_widen f g = binop_to_t' (P.smart_widen f g) (T.smart_widen f g) (U.smart_widen f g)
  let smart_leq f g = binop' (P.smart_leq f g) (T.smart_leq f g) (U.smart_leq f g)
  let update_length newl x = unop_to_t' (P.update_length newl) (T.update_length newl) (U.update_length newl) x

  (* Functions that make use of the configuration flag *)
  let chosen_domain () = get_string "ana.base.arrays.domain"

  let name () = "FlagConfiguredArrayDomain: " ^ match chosen_domain () with
    | "trivial" -> T.name ()
    | "partitioned" -> P.name ()
    | "unroll" -> U.name ()
    | _ -> failwith "FlagConfiguredArrayDomain cannot name an array from set option"

  let bot () =
    to_t @@ match chosen_domain () with
    | "partitioned" -> (Some (P.bot ()), None, None)
    | "trivial" -> (None, Some (T.bot ()), None)
    | "unroll" -> (None, None, Some (U.bot ()))
    | _ -> failwith "FlagConfiguredArrayDomain cannot construct a bot array from set option"

  let top () =
    to_t @@ match chosen_domain () with
    | "partitioned" -> (Some (P.top ()), None, None)
    | "trivial" -> (None, Some (T.top ()), None)
    | "unroll" -> (None, None, Some (U.top ()))
    | _ -> failwith "FlagConfiguredArrayDomain cannot construct a top array from set option"

  let make i v =
    to_t @@ match chosen_domain () with
    | "partitioned" -> (Some (P.make i v), None, None)
    | "trivial" -> (None, Some (T.make i v), None)
    | "unroll" -> (None, None, Some (U.make i v))
    | _ -> failwith "FlagConfiguredArrayDomain cannot construct an array from set option"
end
