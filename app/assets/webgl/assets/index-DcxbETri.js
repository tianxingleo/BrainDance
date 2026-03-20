(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))n(s);new MutationObserver(s=>{for(const r of s)if(r.type==="childList")for(const o of r.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&n(o)}).observe(document,{childList:!0,subtree:!0});function t(s){const r={};return s.integrity&&(r.integrity=s.integrity),s.referrerPolicy&&(r.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?r.credentials="include":s.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function n(s){if(s.ep)return;s.ep=!0;const r=t(s);fetch(s.href,r)}})();function wd(i){const e=Object.create(null);for(const t of i.split(","))e[t]=1;return t=>t in e}const Vt={},To=[],is=()=>{},v0=()=>!1,Fc=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&(i.charCodeAt(2)>122||i.charCodeAt(2)<97),Rd=i=>i.startsWith("onUpdate:"),On=Object.assign,Id=(i,e)=>{const t=i.indexOf(e);t>-1&&i.splice(t,1)},h_=Object.prototype.hasOwnProperty,It=(i,e)=>h_.call(i,e),lt=Array.isArray,Eo=i=>al(i)==="[object Map]",Lc=i=>al(i)==="[object Set]",Gh=i=>al(i)==="[object Date]",pt=i=>typeof i=="function",hn=i=>typeof i=="string",rs=i=>typeof i=="symbol",Nt=i=>i!==null&&typeof i=="object",A0=i=>(Nt(i)||pt(i))&&pt(i.then)&&pt(i.catch),S0=Object.prototype.toString,al=i=>S0.call(i),p_=i=>al(i).slice(8,-1),y0=i=>al(i)==="[object Object]",Dd=i=>hn(i)&&i!=="NaN"&&i[0]!=="-"&&""+parseInt(i,10)===i,ba=wd(",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"),Bc=i=>{const e=Object.create(null);return(t=>e[t]||(e[t]=i(t)))},m_=/-\w/g,ar=Bc(i=>i.replace(m_,e=>e.slice(1).toUpperCase())),g_=/\B([A-Z])/g,hr=Bc(i=>i.replace(g_,"-$1").toLowerCase()),b0=Bc(i=>i.charAt(0).toUpperCase()+i.slice(1)),ru=Bc(i=>i?`on${b0(i)}`:""),nr=(i,e)=>!Object.is(i,e),ec=(i,...e)=>{for(let t=0;t<i.length;t++)i[t](...e)},M0=(i,e,t,n=!1)=>{Object.defineProperty(i,e,{configurable:!0,enumerable:!1,writable:n,value:t})},Pd=i=>{const e=parseFloat(i);return isNaN(e)?i:e};let Wh;const Uc=()=>Wh||(Wh=typeof globalThis<"u"?globalThis:typeof self<"u"?self:typeof window<"u"?window:typeof global<"u"?global:{});function Fd(i){if(lt(i)){const e={};for(let t=0;t<i.length;t++){const n=i[t],s=hn(n)?A_(n):Fd(n);if(s)for(const r in s)e[r]=s[r]}return e}else if(hn(i)||Nt(i))return i}const x_=/;(?![^(]*\))/g,__=/:([^]+)/,v_=/\/\*[^]*?\*\//g;function A_(i){const e={};return i.replace(v_,"").split(x_).forEach(t=>{if(t){const n=t.split(__);n.length>1&&(e[n[0].trim()]=n[1].trim())}}),e}function Or(i){let e="";if(hn(i))e=i;else if(lt(i))for(let t=0;t<i.length;t++){const n=Or(i[t]);n&&(e+=n+" ")}else if(Nt(i))for(const t in i)i[t]&&(e+=t+" ");return e.trim()}const S_="itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly",y_=wd(S_);function C0(i){return!!i||i===""}function b_(i,e){if(i.length!==e.length)return!1;let t=!0;for(let n=0;t&&n<i.length;n++)t=ll(i[n],e[n]);return t}function ll(i,e){if(i===e)return!0;let t=Gh(i),n=Gh(e);if(t||n)return t&&n?i.getTime()===e.getTime():!1;if(t=rs(i),n=rs(e),t||n)return i===e;if(t=lt(i),n=lt(e),t||n)return t&&n?b_(i,e):!1;if(t=Nt(i),n=Nt(e),t||n){if(!t||!n)return!1;const s=Object.keys(i).length,r=Object.keys(e).length;if(s!==r)return!1;for(const o in i){const a=i.hasOwnProperty(o),l=e.hasOwnProperty(o);if(a&&!l||!a&&l||!ll(i[o],e[o]))return!1}}return String(i)===String(e)}function T0(i,e){return i.findIndex(t=>ll(t,e))}const E0=i=>!!(i&&i.__v_isRef===!0),Mn=i=>hn(i)?i:i==null?"":lt(i)||Nt(i)&&(i.toString===S0||!pt(i.toString))?E0(i)?Mn(i.value):JSON.stringify(i,w0,2):String(i),w0=(i,e)=>E0(e)?w0(i,e.value):Eo(e)?{[`Map(${e.size})`]:[...e.entries()].reduce((t,[n,s],r)=>(t[ou(n,r)+" =>"]=s,t),{})}:Lc(e)?{[`Set(${e.size})`]:[...e.values()].map(t=>ou(t))}:rs(e)?ou(e):Nt(e)&&!lt(e)&&!y0(e)?String(e):e,ou=(i,e="")=>{var t;return rs(i)?`Symbol(${(t=i.description)!=null?t:e})`:i};let Jn;class M_{constructor(e=!1){this.detached=e,this._active=!0,this._on=0,this.effects=[],this.cleanups=[],this._isPaused=!1,this.__v_skip=!0,this.parent=Jn,!e&&Jn&&(this.index=(Jn.scopes||(Jn.scopes=[])).push(this)-1)}get active(){return this._active}pause(){if(this._active){this._isPaused=!0;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].pause();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].pause()}}resume(){if(this._active&&this._isPaused){this._isPaused=!1;let e,t;if(this.scopes)for(e=0,t=this.scopes.length;e<t;e++)this.scopes[e].resume();for(e=0,t=this.effects.length;e<t;e++)this.effects[e].resume()}}run(e){if(this._active){const t=Jn;try{return Jn=this,e()}finally{Jn=t}}}on(){++this._on===1&&(this.prevScope=Jn,Jn=this)}off(){this._on>0&&--this._on===0&&(Jn=this.prevScope,this.prevScope=void 0)}stop(e){if(this._active){this._active=!1;let t,n;for(t=0,n=this.effects.length;t<n;t++)this.effects[t].stop();for(this.effects.length=0,t=0,n=this.cleanups.length;t<n;t++)this.cleanups[t]();if(this.cleanups.length=0,this.scopes){for(t=0,n=this.scopes.length;t<n;t++)this.scopes[t].stop(!0);this.scopes.length=0}if(!this.detached&&this.parent&&!e){const s=this.parent.scopes.pop();s&&s!==this&&(this.parent.scopes[this.index]=s,s.index=this.index)}this.parent=void 0}}}function C_(){return Jn}let Wt;const au=new WeakSet;class R0{constructor(e){this.fn=e,this.deps=void 0,this.depsTail=void 0,this.flags=5,this.next=void 0,this.cleanup=void 0,this.scheduler=void 0,Jn&&Jn.active&&Jn.effects.push(this)}pause(){this.flags|=64}resume(){this.flags&64&&(this.flags&=-65,au.has(this)&&(au.delete(this),this.trigger()))}notify(){this.flags&2&&!(this.flags&32)||this.flags&8||D0(this)}run(){if(!(this.flags&1))return this.fn();this.flags|=2,Xh(this),P0(this);const e=Wt,t=Hi;Wt=this,Hi=!0;try{return this.fn()}finally{F0(this),Wt=e,Hi=t,this.flags&=-3}}stop(){if(this.flags&1){for(let e=this.deps;e;e=e.nextDep)Ud(e);this.deps=this.depsTail=void 0,Xh(this),this.onStop&&this.onStop(),this.flags&=-2}}trigger(){this.flags&64?au.add(this):this.scheduler?this.scheduler():this.runIfDirty()}runIfDirty(){hf(this)&&this.run()}get dirty(){return hf(this)}}let I0=0,Ma,Ca;function D0(i,e=!1){if(i.flags|=8,e){i.next=Ca,Ca=i;return}i.next=Ma,Ma=i}function Ld(){I0++}function Bd(){if(--I0>0)return;if(Ca){let e=Ca;for(Ca=void 0;e;){const t=e.next;e.next=void 0,e.flags&=-9,e=t}}let i;for(;Ma;){let e=Ma;for(Ma=void 0;e;){const t=e.next;if(e.next=void 0,e.flags&=-9,e.flags&1)try{e.trigger()}catch(n){i||(i=n)}e=t}}if(i)throw i}function P0(i){for(let e=i.deps;e;e=e.nextDep)e.version=-1,e.prevActiveLink=e.dep.activeLink,e.dep.activeLink=e}function F0(i){let e,t=i.depsTail,n=t;for(;n;){const s=n.prevDep;n.version===-1?(n===t&&(t=s),Ud(n),T_(n)):e=n,n.dep.activeLink=n.prevActiveLink,n.prevActiveLink=void 0,n=s}i.deps=e,i.depsTail=t}function hf(i){for(let e=i.deps;e;e=e.nextDep)if(e.dep.version!==e.version||e.dep.computed&&(L0(e.dep.computed)||e.dep.version!==e.version))return!0;return!!i._dirty}function L0(i){if(i.flags&4&&!(i.flags&16)||(i.flags&=-17,i.globalVersion===ka)||(i.globalVersion=ka,!i.isSSR&&i.flags&128&&(!i.deps&&!i._dirty||!hf(i))))return;i.flags|=2;const e=i.dep,t=Wt,n=Hi;Wt=i,Hi=!0;try{P0(i);const s=i.fn(i._value);(e.version===0||nr(s,i._value))&&(i.flags|=128,i._value=s,e.version++)}catch(s){throw e.version++,s}finally{Wt=t,Hi=n,F0(i),i.flags&=-3}}function Ud(i,e=!1){const{dep:t,prevSub:n,nextSub:s}=i;if(n&&(n.nextSub=s,i.prevSub=void 0),s&&(s.prevSub=n,i.nextSub=void 0),t.subs===i&&(t.subs=n,!n&&t.computed)){t.computed.flags&=-5;for(let r=t.computed.deps;r;r=r.nextDep)Ud(r,!0)}!e&&!--t.sc&&t.map&&t.map.delete(t.key)}function T_(i){const{prevDep:e,nextDep:t}=i;e&&(e.nextDep=t,i.prevDep=void 0),t&&(t.prevDep=e,i.nextDep=void 0)}let Hi=!0;const B0=[];function Fs(){B0.push(Hi),Hi=!1}function Ls(){const i=B0.pop();Hi=i===void 0?!0:i}function Xh(i){const{cleanup:e}=i;if(i.cleanup=void 0,e){const t=Wt;Wt=void 0;try{e()}finally{Wt=t}}}let ka=0;class E_{constructor(e,t){this.sub=e,this.dep=t,this.version=t.version,this.nextDep=this.prevDep=this.nextSub=this.prevSub=this.prevActiveLink=void 0}}class Od{constructor(e){this.computed=e,this.version=0,this.activeLink=void 0,this.subs=void 0,this.map=void 0,this.key=void 0,this.sc=0,this.__v_skip=!0}track(e){if(!Wt||!Hi||Wt===this.computed)return;let t=this.activeLink;if(t===void 0||t.sub!==Wt)t=this.activeLink=new E_(Wt,this),Wt.deps?(t.prevDep=Wt.depsTail,Wt.depsTail.nextDep=t,Wt.depsTail=t):Wt.deps=Wt.depsTail=t,U0(t);else if(t.version===-1&&(t.version=this.version,t.nextDep)){const n=t.nextDep;n.prevDep=t.prevDep,t.prevDep&&(t.prevDep.nextDep=n),t.prevDep=Wt.depsTail,t.nextDep=void 0,Wt.depsTail.nextDep=t,Wt.depsTail=t,Wt.deps===t&&(Wt.deps=n)}return t}trigger(e){this.version++,ka++,this.notify(e)}notify(e){Ld();try{for(let t=this.subs;t;t=t.prevSub)t.sub.notify()&&t.sub.dep.notify()}finally{Bd()}}}function U0(i){if(i.dep.sc++,i.sub.flags&4){const e=i.dep.computed;if(e&&!i.dep.subs){e.flags|=20;for(let n=e.deps;n;n=n.nextDep)U0(n)}const t=i.dep.subs;t!==i&&(i.prevSub=t,t&&(t.nextSub=i)),i.dep.subs=i}}const pf=new WeakMap,Hr=Symbol(""),mf=Symbol(""),Ha=Symbol("");function Pn(i,e,t){if(Hi&&Wt){let n=pf.get(i);n||pf.set(i,n=new Map);let s=n.get(t);s||(n.set(t,s=new Od),s.map=n,s.key=t),s.track()}}function Es(i,e,t,n,s,r){const o=pf.get(i);if(!o){ka++;return}const a=l=>{l&&l.trigger()};if(Ld(),e==="clear")o.forEach(a);else{const l=lt(i),c=l&&Dd(t);if(l&&t==="length"){const u=Number(n);o.forEach((f,d)=>{(d==="length"||d===Ha||!rs(d)&&d>=u)&&a(f)})}else switch((t!==void 0||o.has(void 0))&&a(o.get(t)),c&&a(o.get(Ha)),e){case"add":l?c&&a(o.get("length")):(a(o.get(Hr)),Eo(i)&&a(o.get(mf)));break;case"delete":l||(a(o.get(Hr)),Eo(i)&&a(o.get(mf)));break;case"set":Eo(i)&&a(o.get(Hr));break}}Bd()}function Jr(i){const e=Rt(i);return e===i?e:(Pn(e,"iterate",Ha),Fi(i)?e:e.map(Vi))}function Oc(i){return Pn(i=Rt(i),"iterate",Ha),i}function Ys(i,e){return Bs(i)?zo(Vr(i)?Vi(e):e):Vi(e)}const w_={__proto__:null,[Symbol.iterator](){return lu(this,Symbol.iterator,i=>Ys(this,i))},concat(...i){return Jr(this).concat(...i.map(e=>lt(e)?Jr(e):e))},entries(){return lu(this,"entries",i=>(i[1]=Ys(this,i[1]),i))},every(i,e){return ds(this,"every",i,e,void 0,arguments)},filter(i,e){return ds(this,"filter",i,e,t=>t.map(n=>Ys(this,n)),arguments)},find(i,e){return ds(this,"find",i,e,t=>Ys(this,t),arguments)},findIndex(i,e){return ds(this,"findIndex",i,e,void 0,arguments)},findLast(i,e){return ds(this,"findLast",i,e,t=>Ys(this,t),arguments)},findLastIndex(i,e){return ds(this,"findLastIndex",i,e,void 0,arguments)},forEach(i,e){return ds(this,"forEach",i,e,void 0,arguments)},includes(...i){return cu(this,"includes",i)},indexOf(...i){return cu(this,"indexOf",i)},join(i){return Jr(this).join(i)},lastIndexOf(...i){return cu(this,"lastIndexOf",i)},map(i,e){return ds(this,"map",i,e,void 0,arguments)},pop(){return ca(this,"pop")},push(...i){return ca(this,"push",i)},reduce(i,...e){return qh(this,"reduce",i,e)},reduceRight(i,...e){return qh(this,"reduceRight",i,e)},shift(){return ca(this,"shift")},some(i,e){return ds(this,"some",i,e,void 0,arguments)},splice(...i){return ca(this,"splice",i)},toReversed(){return Jr(this).toReversed()},toSorted(i){return Jr(this).toSorted(i)},toSpliced(...i){return Jr(this).toSpliced(...i)},unshift(...i){return ca(this,"unshift",i)},values(){return lu(this,"values",i=>Ys(this,i))}};function lu(i,e,t){const n=Oc(i),s=n[e]();return n!==i&&!Fi(i)&&(s._next=s.next,s.next=()=>{const r=s._next();return r.done||(r.value=t(r.value)),r}),s}const R_=Array.prototype;function ds(i,e,t,n,s,r){const o=Oc(i),a=o!==i&&!Fi(i),l=o[e];if(l!==R_[e]){const f=l.apply(i,r);return a?Vi(f):f}let c=t;o!==i&&(a?c=function(f,d){return t.call(this,Ys(i,f),d,i)}:t.length>2&&(c=function(f,d){return t.call(this,f,d,i)}));const u=l.call(o,c,n);return a&&s?s(u):u}function qh(i,e,t,n){const s=Oc(i);let r=t;return s!==i&&(Fi(i)?t.length>3&&(r=function(o,a,l){return t.call(this,o,a,l,i)}):r=function(o,a,l){return t.call(this,o,Ys(i,a),l,i)}),s[e](r,...n)}function cu(i,e,t){const n=Rt(i);Pn(n,"iterate",Ha);const s=n[e](...t);return(s===-1||s===!1)&&Hd(t[0])?(t[0]=Rt(t[0]),n[e](...t)):s}function ca(i,e,t=[]){Fs(),Ld();const n=Rt(i)[e].apply(i,t);return Bd(),Ls(),n}const I_=wd("__proto__,__v_isRef,__isVue"),O0=new Set(Object.getOwnPropertyNames(Symbol).filter(i=>i!=="arguments"&&i!=="caller").map(i=>Symbol[i]).filter(rs));function D_(i){rs(i)||(i=String(i));const e=Rt(this);return Pn(e,"has",i),e.hasOwnProperty(i)}class N0{constructor(e=!1,t=!1){this._isReadonly=e,this._isShallow=t}get(e,t,n){if(t==="__v_skip")return e.__v_skip;const s=this._isReadonly,r=this._isShallow;if(t==="__v_isReactive")return!s;if(t==="__v_isReadonly")return s;if(t==="__v_isShallow")return r;if(t==="__v_raw")return n===(s?r?H_:V0:r?H0:k0).get(e)||Object.getPrototypeOf(e)===Object.getPrototypeOf(n)?e:void 0;const o=lt(e);if(!s){let l;if(o&&(l=w_[t]))return l;if(t==="hasOwnProperty")return D_}const a=Reflect.get(e,t,Ln(e)?e:n);if((rs(t)?O0.has(t):I_(t))||(s||Pn(e,"get",t),r))return a;if(Ln(a)){const l=o&&Dd(t)?a:a.value;return s&&Nt(l)?xf(l):l}return Nt(a)?s?xf(a):zd(a):a}}class z0 extends N0{constructor(e=!1){super(!1,e)}set(e,t,n,s){let r=e[t];const o=lt(e)&&Dd(t);if(!this._isShallow){const c=Bs(r);if(!Fi(n)&&!Bs(n)&&(r=Rt(r),n=Rt(n)),!o&&Ln(r)&&!Ln(n))return c||(r.value=n),!0}const a=o?Number(t)<e.length:It(e,t),l=Reflect.set(e,t,n,Ln(e)?e:s);return e===Rt(s)&&(a?nr(n,r)&&Es(e,"set",t,n):Es(e,"add",t,n)),l}deleteProperty(e,t){const n=It(e,t);e[t];const s=Reflect.deleteProperty(e,t);return s&&n&&Es(e,"delete",t,void 0),s}has(e,t){const n=Reflect.has(e,t);return(!rs(t)||!O0.has(t))&&Pn(e,"has",t),n}ownKeys(e){return Pn(e,"iterate",lt(e)?"length":Hr),Reflect.ownKeys(e)}}class P_ extends N0{constructor(e=!1){super(!0,e)}set(e,t){return!0}deleteProperty(e,t){return!0}}const F_=new z0,L_=new P_,B_=new z0(!0);const gf=i=>i,_l=i=>Reflect.getPrototypeOf(i);function U_(i,e,t){return function(...n){const s=this.__v_raw,r=Rt(s),o=Eo(r),a=i==="entries"||i===Symbol.iterator&&o,l=i==="keys"&&o,c=s[i](...n),u=t?gf:e?zo:Vi;return!e&&Pn(r,"iterate",l?mf:Hr),On(Object.create(c),{next(){const{value:f,done:d}=c.next();return d?{value:f,done:d}:{value:a?[u(f[0]),u(f[1])]:u(f),done:d}}})}}function vl(i){return function(...e){return i==="delete"?!1:i==="clear"?void 0:this}}function O_(i,e){const t={get(s){const r=this.__v_raw,o=Rt(r),a=Rt(s);i||(nr(s,a)&&Pn(o,"get",s),Pn(o,"get",a));const{has:l}=_l(o),c=e?gf:i?zo:Vi;if(l.call(o,s))return c(r.get(s));if(l.call(o,a))return c(r.get(a));r!==o&&r.get(s)},get size(){const s=this.__v_raw;return!i&&Pn(Rt(s),"iterate",Hr),s.size},has(s){const r=this.__v_raw,o=Rt(r),a=Rt(s);return i||(nr(s,a)&&Pn(o,"has",s),Pn(o,"has",a)),s===a?r.has(s):r.has(s)||r.has(a)},forEach(s,r){const o=this,a=o.__v_raw,l=Rt(a),c=e?gf:i?zo:Vi;return!i&&Pn(l,"iterate",Hr),a.forEach((u,f)=>s.call(r,c(u),c(f),o))}};return On(t,i?{add:vl("add"),set:vl("set"),delete:vl("delete"),clear:vl("clear")}:{add(s){!e&&!Fi(s)&&!Bs(s)&&(s=Rt(s));const r=Rt(this);return _l(r).has.call(r,s)||(r.add(s),Es(r,"add",s,s)),this},set(s,r){!e&&!Fi(r)&&!Bs(r)&&(r=Rt(r));const o=Rt(this),{has:a,get:l}=_l(o);let c=a.call(o,s);c||(s=Rt(s),c=a.call(o,s));const u=l.call(o,s);return o.set(s,r),c?nr(r,u)&&Es(o,"set",s,r):Es(o,"add",s,r),this},delete(s){const r=Rt(this),{has:o,get:a}=_l(r);let l=o.call(r,s);l||(s=Rt(s),l=o.call(r,s)),a&&a.call(r,s);const c=r.delete(s);return l&&Es(r,"delete",s,void 0),c},clear(){const s=Rt(this),r=s.size!==0,o=s.clear();return r&&Es(s,"clear",void 0,void 0),o}}),["keys","values","entries",Symbol.iterator].forEach(s=>{t[s]=U_(s,i,e)}),t}function Nd(i,e){const t=O_(i,e);return(n,s,r)=>s==="__v_isReactive"?!i:s==="__v_isReadonly"?i:s==="__v_raw"?n:Reflect.get(It(t,s)&&s in n?t:n,s,r)}const N_={get:Nd(!1,!1)},z_={get:Nd(!1,!0)},k_={get:Nd(!0,!1)};const k0=new WeakMap,H0=new WeakMap,V0=new WeakMap,H_=new WeakMap;function V_(i){switch(i){case"Object":case"Array":return 1;case"Map":case"Set":case"WeakMap":case"WeakSet":return 2;default:return 0}}function G_(i){return i.__v_skip||!Object.isExtensible(i)?0:V_(p_(i))}function zd(i){return Bs(i)?i:kd(i,!1,F_,N_,k0)}function W_(i){return kd(i,!1,B_,z_,H0)}function xf(i){return kd(i,!0,L_,k_,V0)}function kd(i,e,t,n,s){if(!Nt(i)||i.__v_raw&&!(e&&i.__v_isReactive))return i;const r=G_(i);if(r===0)return i;const o=s.get(i);if(o)return o;const a=new Proxy(i,r===2?n:t);return s.set(i,a),a}function Vr(i){return Bs(i)?Vr(i.__v_raw):!!(i&&i.__v_isReactive)}function Bs(i){return!!(i&&i.__v_isReadonly)}function Fi(i){return!!(i&&i.__v_isShallow)}function Hd(i){return i?!!i.__v_raw:!1}function Rt(i){const e=i&&i.__v_raw;return e?Rt(e):i}function X_(i){return!It(i,"__v_skip")&&Object.isExtensible(i)&&M0(i,"__v_skip",!0),i}const Vi=i=>Nt(i)?zd(i):i,zo=i=>Nt(i)?xf(i):i;function Ln(i){return i?i.__v_isRef===!0:!1}function Ft(i){return q_(i,!1)}function q_(i,e){return Ln(i)?i:new Y_(i,e)}class Y_{constructor(e,t){this.dep=new Od,this.__v_isRef=!0,this.__v_isShallow=!1,this._rawValue=t?e:Rt(e),this._value=t?e:Vi(e),this.__v_isShallow=t}get value(){return this.dep.track(),this._value}set value(e){const t=this._rawValue,n=this.__v_isShallow||Fi(e)||Bs(e);e=n?e:Rt(e),nr(e,t)&&(this._rawValue=e,this._value=n?e:Vi(e),this.dep.trigger())}}function _f(i){return Ln(i)?i.value:i}const Q_={get:(i,e,t)=>e==="__v_raw"?i:_f(Reflect.get(i,e,t)),set:(i,e,t,n)=>{const s=i[e];return Ln(s)&&!Ln(t)?(s.value=t,!0):Reflect.set(i,e,t,n)}};function G0(i){return Vr(i)?i:new Proxy(i,Q_)}class K_{constructor(e,t,n){this.fn=e,this.setter=t,this._value=void 0,this.dep=new Od(this),this.__v_isRef=!0,this.deps=void 0,this.depsTail=void 0,this.flags=16,this.globalVersion=ka-1,this.next=void 0,this.effect=this,this.__v_isReadonly=!t,this.isSSR=n}notify(){if(this.flags|=16,!(this.flags&8)&&Wt!==this)return D0(this,!0),!0}get value(){const e=this.dep.track();return L0(this),e&&(e.version=this.dep.version),this._value}set value(e){this.setter&&this.setter(e)}}function j_(i,e,t=!1){let n,s;return pt(i)?n=i:(n=i.get,s=i.set),new K_(n,s,t)}const Al={},dc=new WeakMap;let Rr;function $_(i,e=!1,t=Rr){if(t){let n=dc.get(t);n||dc.set(t,n=[]),n.push(i)}}function Z_(i,e,t=Vt){const{immediate:n,deep:s,once:r,scheduler:o,augmentJob:a,call:l}=t,c=A=>s?A:Fi(A)||s===!1||s===0?ws(A,1):ws(A);let u,f,d,h,x=!1,p=!1;if(Ln(i)?(f=()=>i.value,x=Fi(i)):Vr(i)?(f=()=>c(i),x=!0):lt(i)?(p=!0,x=i.some(A=>Vr(A)||Fi(A)),f=()=>i.map(A=>{if(Ln(A))return A.value;if(Vr(A))return c(A);if(pt(A))return l?l(A,2):A()})):pt(i)?e?f=l?()=>l(i,2):i:f=()=>{if(d){Fs();try{d()}finally{Ls()}}const A=Rr;Rr=u;try{return l?l(i,3,[h]):i(h)}finally{Rr=A}}:f=is,e&&s){const A=f,S=s===!0?1/0:s;f=()=>ws(A(),S)}const g=C_(),m=()=>{u.stop(),g&&g.active&&Id(g.effects,u)};if(r&&e){const A=e;e=(...S)=>{A(...S),m()}}let _=p?new Array(i.length).fill(Al):Al;const v=A=>{if(!(!(u.flags&1)||!u.dirty&&!A))if(e){const S=u.run();if(s||x||(p?S.some((y,M)=>nr(y,_[M])):nr(S,_))){d&&d();const y=Rr;Rr=u;try{const M=[S,_===Al?void 0:p&&_[0]===Al?[]:_,h];_=S,l?l(e,3,M):e(...M)}finally{Rr=y}}}else u.run()};return a&&a(v),u=new R0(f),u.scheduler=o?()=>o(v,!1):v,h=A=>$_(A,!1,u),d=u.onStop=()=>{const A=dc.get(u);if(A){if(l)l(A,4);else for(const S of A)S();dc.delete(u)}},e?n?v(!0):_=u.run():o?o(v.bind(null,!0),!0):u.run(),m.pause=u.pause.bind(u),m.resume=u.resume.bind(u),m.stop=m,m}function ws(i,e=1/0,t){if(e<=0||!Nt(i)||i.__v_skip||(t=t||new Map,(t.get(i)||0)>=e))return i;if(t.set(i,e),e--,Ln(i))ws(i.value,e,t);else if(lt(i))for(let n=0;n<i.length;n++)ws(i[n],e,t);else if(Lc(i)||Eo(i))i.forEach(n=>{ws(n,e,t)});else if(y0(i)){for(const n in i)ws(i[n],e,t);for(const n of Object.getOwnPropertySymbols(i))Object.prototype.propertyIsEnumerable.call(i,n)&&ws(i[n],e,t)}return i}function cl(i,e,t,n){try{return n?i(...n):i()}catch(s){Nc(s,e,t)}}function os(i,e,t,n){if(pt(i)){const s=cl(i,e,t,n);return s&&A0(s)&&s.catch(r=>{Nc(r,e,t)}),s}if(lt(i)){const s=[];for(let r=0;r<i.length;r++)s.push(os(i[r],e,t,n));return s}}function Nc(i,e,t,n=!0){const s=e?e.vnode:null,{errorHandler:r,throwUnhandledErrorInProduction:o}=e&&e.appContext.config||Vt;if(e){let a=e.parent;const l=e.proxy,c=`https://vuejs.org/error-reference/#runtime-${t}`;for(;a;){const u=a.ec;if(u){for(let f=0;f<u.length;f++)if(u[f](i,l,c)===!1)return}a=a.parent}if(r){Fs(),cl(r,null,10,[i,l,c]),Ls();return}}J_(i,t,s,n,o)}function J_(i,e,t,n=!0,s=!1){if(s)throw i;console.error(i)}const Gn=[];let Yi=-1;const wo=[];let Qs=null,Ao=0;const W0=Promise.resolve();let hc=null;function ev(i){const e=hc||W0;return i?e.then(this?i.bind(this):i):e}function tv(i){let e=Yi+1,t=Gn.length;for(;e<t;){const n=e+t>>>1,s=Gn[n],r=Va(s);r<i||r===i&&s.flags&2?e=n+1:t=n}return e}function Vd(i){if(!(i.flags&1)){const e=Va(i),t=Gn[Gn.length-1];!t||!(i.flags&2)&&e>=Va(t)?Gn.push(i):Gn.splice(tv(e),0,i),i.flags|=1,X0()}}function X0(){hc||(hc=W0.then(Y0))}function nv(i){lt(i)?wo.push(...i):Qs&&i.id===-1?Qs.splice(Ao+1,0,i):i.flags&1||(wo.push(i),i.flags|=1),X0()}function Yh(i,e,t=Yi+1){for(;t<Gn.length;t++){const n=Gn[t];if(n&&n.flags&2){if(i&&n.id!==i.uid)continue;Gn.splice(t,1),t--,n.flags&4&&(n.flags&=-2),n(),n.flags&4||(n.flags&=-2)}}}function q0(i){if(wo.length){const e=[...new Set(wo)].sort((t,n)=>Va(t)-Va(n));if(wo.length=0,Qs){Qs.push(...e);return}for(Qs=e,Ao=0;Ao<Qs.length;Ao++){const t=Qs[Ao];t.flags&4&&(t.flags&=-2),t.flags&8||t(),t.flags&=-2}Qs=null,Ao=0}}const Va=i=>i.id==null?i.flags&2?-1:1/0:i.id;function Y0(i){try{for(Yi=0;Yi<Gn.length;Yi++){const e=Gn[Yi];e&&!(e.flags&8)&&(e.flags&4&&(e.flags&=-2),cl(e,e.i,e.i?15:14),e.flags&4||(e.flags&=-2))}}finally{for(;Yi<Gn.length;Yi++){const e=Gn[Yi];e&&(e.flags&=-2)}Yi=-1,Gn.length=0,q0(),hc=null,(Gn.length||wo.length)&&Y0()}}let wi=null,Q0=null;function pc(i){const e=wi;return wi=i,Q0=i&&i.type.__scopeId||null,e}function iv(i,e=wi,t){if(!e||i._n)return i;const n=(...s)=>{n._d&&sp(-1);const r=pc(e);let o;try{o=i(...s)}finally{pc(r),n._d&&sp(1)}return o};return n._n=!0,n._c=!0,n._d=!0,n}function xr(i,e){if(wi===null)return i;const t=Vc(wi),n=i.dirs||(i.dirs=[]);for(let s=0;s<e.length;s++){let[r,o,a,l=Vt]=e[s];r&&(pt(r)&&(r={mounted:r,updated:r}),r.deep&&ws(o),n.push({dir:r,instance:t,value:o,oldValue:void 0,arg:a,modifiers:l}))}return i}function _r(i,e,t,n){const s=i.dirs,r=e&&e.dirs;for(let o=0;o<s.length;o++){const a=s[o];r&&(a.oldValue=r[o].value);let l=a.dir[n];l&&(Fs(),os(l,t,8,[i.el,a,i,e]),Ls())}}function sv(i,e){if(Xn){let t=Xn.provides;const n=Xn.parent&&Xn.parent.provides;n===t&&(t=Xn.provides=Object.create(n)),t[i]=e}}function tc(i,e,t=!1){const n=sA();if(n||Ro){let s=Ro?Ro._context.provides:n?n.parent==null||n.ce?n.vnode.appContext&&n.vnode.appContext.provides:n.parent.provides:void 0;if(s&&i in s)return s[i];if(arguments.length>1)return t&&pt(e)?e.call(n&&n.proxy):e}}const rv=Symbol.for("v-scx"),ov=()=>tc(rv);function uu(i,e,t){return K0(i,e,t)}function K0(i,e,t=Vt){const{immediate:n,deep:s,flush:r,once:o}=t,a=On({},t),l=e&&n||!e&&r!=="post";let c;if(Wa){if(r==="sync"){const h=ov();c=h.__watcherHandles||(h.__watcherHandles=[])}else if(!l){const h=()=>{};return h.stop=is,h.resume=is,h.pause=is,h}}const u=Xn;a.call=(h,x,p)=>os(h,u,x,p);let f=!1;r==="post"?a.scheduler=h=>{Zn(h,u&&u.suspense)}:r!=="sync"&&(f=!0,a.scheduler=(h,x)=>{x?h():Vd(h)}),a.augmentJob=h=>{e&&(h.flags|=4),f&&(h.flags|=2,u&&(h.id=u.uid,h.i=u))};const d=Z_(i,e,a);return Wa&&(c?c.push(d):l&&d()),d}function av(i,e,t){const n=this.proxy,s=hn(i)?i.includes(".")?j0(n,i):()=>n[i]:i.bind(n,n);let r;pt(e)?r=e:(r=e.handler,t=e);const o=ul(this),a=K0(s,r.bind(n),t);return o(),a}function j0(i,e){const t=e.split(".");return()=>{let n=i;for(let s=0;s<t.length&&n;s++)n=n[t[s]];return n}}const lv=Symbol("_vte"),cv=i=>i.__isTeleport,uv=Symbol("_leaveCb");function Gd(i,e){i.shapeFlag&6&&i.component?(i.transition=e,Gd(i.component.subTree,e)):i.shapeFlag&128?(i.ssContent.transition=e.clone(i.ssContent),i.ssFallback.transition=e.clone(i.ssFallback)):i.transition=e}function $0(i){i.ids=[i.ids[0]+i.ids[2]+++"-",0,0]}function Qh(i,e){let t;return!!((t=Object.getOwnPropertyDescriptor(i,e))&&!t.configurable)}const mc=new WeakMap;function Ta(i,e,t,n,s=!1){if(lt(i)){i.forEach((p,g)=>Ta(p,e&&(lt(e)?e[g]:e),t,n,s));return}if(Ea(n)&&!s){n.shapeFlag&512&&n.type.__asyncResolved&&n.component.subTree.component&&Ta(i,e,t,n.component.subTree);return}const r=n.shapeFlag&4?Vc(n.component):n.el,o=s?null:r,{i:a,r:l}=i,c=e&&e.r,u=a.refs===Vt?a.refs={}:a.refs,f=a.setupState,d=Rt(f),h=f===Vt?v0:p=>Qh(u,p)?!1:It(d,p),x=(p,g)=>!(g&&Qh(u,g));if(c!=null&&c!==l){if(Kh(e),hn(c))u[c]=null,h(c)&&(f[c]=null);else if(Ln(c)){const p=e;x(c,p.k)&&(c.value=null),p.k&&(u[p.k]=null)}}if(pt(l))cl(l,a,12,[o,u]);else{const p=hn(l),g=Ln(l);if(p||g){const m=()=>{if(i.f){const _=p?h(l)?f[l]:u[l]:x()||!i.k?l.value:u[i.k];if(s)lt(_)&&Id(_,r);else if(lt(_))_.includes(r)||_.push(r);else if(p)u[l]=[r],h(l)&&(f[l]=u[l]);else{const v=[r];x(l,i.k)&&(l.value=v),i.k&&(u[i.k]=v)}}else p?(u[l]=o,h(l)&&(f[l]=o)):g&&(x(l,i.k)&&(l.value=o),i.k&&(u[i.k]=o))};if(o){const _=()=>{m(),mc.delete(i)};_.id=-1,mc.set(i,_),Zn(_,t)}else Kh(i),m()}}}function Kh(i){const e=mc.get(i);e&&(e.flags|=8,mc.delete(i))}Uc().requestIdleCallback;Uc().cancelIdleCallback;const Ea=i=>!!i.type.__asyncLoader,Z0=i=>i.type.__isKeepAlive;function fv(i,e){J0(i,"a",e)}function dv(i,e){J0(i,"da",e)}function J0(i,e,t=Xn){const n=i.__wdc||(i.__wdc=()=>{let s=t;for(;s;){if(s.isDeactivated)return;s=s.parent}return i()});if(zc(e,n,t),t){let s=t.parent;for(;s&&s.parent;)Z0(s.parent.vnode)&&hv(n,e,t,s),s=s.parent}}function hv(i,e,t,n){const s=zc(e,i,n,!0);ng(()=>{Id(n[e],s)},t)}function zc(i,e,t=Xn,n=!1){if(t){const s=t[i]||(t[i]=[]),r=e.__weh||(e.__weh=(...o)=>{Fs();const a=ul(t),l=os(e,t,i,o);return a(),Ls(),l});return n?s.unshift(r):s.push(r),r}}const Ns=i=>(e,t=Xn)=>{(!Wa||i==="sp")&&zc(i,(...n)=>e(...n),t)},pv=Ns("bm"),eg=Ns("m"),mv=Ns("bu"),gv=Ns("u"),tg=Ns("bum"),ng=Ns("um"),xv=Ns("sp"),_v=Ns("rtg"),vv=Ns("rtc");function Av(i,e=Xn){zc("ec",i,e)}const Sv=Symbol.for("v-ndc");function yv(i,e,t,n){let s;const r=t,o=lt(i);if(o||hn(i)){const a=o&&Vr(i);let l=!1,c=!1;a&&(l=!Fi(i),c=Bs(i),i=Oc(i)),s=new Array(i.length);for(let u=0,f=i.length;u<f;u++)s[u]=e(l?c?zo(Vi(i[u])):Vi(i[u]):i[u],u,void 0,r)}else if(typeof i=="number"){s=new Array(i);for(let a=0;a<i;a++)s[a]=e(a+1,a,void 0,r)}else if(Nt(i))if(i[Symbol.iterator])s=Array.from(i,(a,l)=>e(a,l,void 0,r));else{const a=Object.keys(i);s=new Array(a.length);for(let l=0,c=a.length;l<c;l++){const u=a[l];s[l]=e(i[u],u,l,r)}}else s=[];return s}const vf=i=>i?bg(i)?Vc(i):vf(i.parent):null,wa=On(Object.create(null),{$:i=>i,$el:i=>i.vnode.el,$data:i=>i.data,$props:i=>i.props,$attrs:i=>i.attrs,$slots:i=>i.slots,$refs:i=>i.refs,$parent:i=>vf(i.parent),$root:i=>vf(i.root),$host:i=>i.ce,$emit:i=>i.emit,$options:i=>sg(i),$forceUpdate:i=>i.f||(i.f=()=>{Vd(i.update)}),$nextTick:i=>i.n||(i.n=ev.bind(i.proxy)),$watch:i=>av.bind(i)}),fu=(i,e)=>i!==Vt&&!i.__isScriptSetup&&It(i,e),bv={get({_:i},e){if(e==="__v_skip")return!0;const{ctx:t,setupState:n,data:s,props:r,accessCache:o,type:a,appContext:l}=i;if(e[0]!=="$"){const d=o[e];if(d!==void 0)switch(d){case 1:return n[e];case 2:return s[e];case 4:return t[e];case 3:return r[e]}else{if(fu(n,e))return o[e]=1,n[e];if(s!==Vt&&It(s,e))return o[e]=2,s[e];if(It(r,e))return o[e]=3,r[e];if(t!==Vt&&It(t,e))return o[e]=4,t[e];Af&&(o[e]=0)}}const c=wa[e];let u,f;if(c)return e==="$attrs"&&Pn(i.attrs,"get",""),c(i);if((u=a.__cssModules)&&(u=u[e]))return u;if(t!==Vt&&It(t,e))return o[e]=4,t[e];if(f=l.config.globalProperties,It(f,e))return f[e]},set({_:i},e,t){const{data:n,setupState:s,ctx:r}=i;return fu(s,e)?(s[e]=t,!0):n!==Vt&&It(n,e)?(n[e]=t,!0):It(i.props,e)||e[0]==="$"&&e.slice(1)in i?!1:(r[e]=t,!0)},has({_:{data:i,setupState:e,accessCache:t,ctx:n,appContext:s,props:r,type:o}},a){let l;return!!(t[a]||i!==Vt&&a[0]!=="$"&&It(i,a)||fu(e,a)||It(r,a)||It(n,a)||It(wa,a)||It(s.config.globalProperties,a)||(l=o.__cssModules)&&l[a])},defineProperty(i,e,t){return t.get!=null?i._.accessCache[e]=0:It(t,"value")&&this.set(i,e,t.value,null),Reflect.defineProperty(i,e,t)}};function jh(i){return lt(i)?i.reduce((e,t)=>(e[t]=null,e),{}):i}let Af=!0;function Mv(i){const e=sg(i),t=i.proxy,n=i.ctx;Af=!1,e.beforeCreate&&$h(e.beforeCreate,i,"bc");const{data:s,computed:r,methods:o,watch:a,provide:l,inject:c,created:u,beforeMount:f,mounted:d,beforeUpdate:h,updated:x,activated:p,deactivated:g,beforeDestroy:m,beforeUnmount:_,destroyed:v,unmounted:A,render:S,renderTracked:y,renderTriggered:M,errorCaptured:E,serverPrefetch:b,expose:C,inheritAttrs:P,components:B,directives:N,filters:k}=e;if(c&&Cv(c,n,null),o)for(const q in o){const G=o[q];pt(G)&&(n[q]=G.bind(t))}if(s){const q=s.call(t,t);Nt(q)&&(i.data=zd(q))}if(Af=!0,r)for(const q in r){const G=r[q],j=pt(G)?G.bind(t,t):pt(G.get)?G.get.bind(t,t):is,ce=!pt(G)&&pt(G.set)?G.set.bind(t):is,ye=Ir({get:j,set:ce});Object.defineProperty(n,q,{enumerable:!0,configurable:!0,get:()=>ye.value,set:Ie=>ye.value=Ie})}if(a)for(const q in a)ig(a[q],n,t,q);if(l){const q=pt(l)?l.call(t):l;Reflect.ownKeys(q).forEach(G=>{sv(G,q[G])})}u&&$h(u,i,"c");function F(q,G){lt(G)?G.forEach(j=>q(j.bind(t))):G&&q(G.bind(t))}if(F(pv,f),F(eg,d),F(mv,h),F(gv,x),F(fv,p),F(dv,g),F(Av,E),F(vv,y),F(_v,M),F(tg,_),F(ng,A),F(xv,b),lt(C))if(C.length){const q=i.exposed||(i.exposed={});C.forEach(G=>{Object.defineProperty(q,G,{get:()=>t[G],set:j=>t[G]=j,enumerable:!0})})}else i.exposed||(i.exposed={});S&&i.render===is&&(i.render=S),P!=null&&(i.inheritAttrs=P),B&&(i.components=B),N&&(i.directives=N),b&&$0(i)}function Cv(i,e,t=is){lt(i)&&(i=Sf(i));for(const n in i){const s=i[n];let r;Nt(s)?"default"in s?r=tc(s.from||n,s.default,!0):r=tc(s.from||n):r=tc(s),Ln(r)?Object.defineProperty(e,n,{enumerable:!0,configurable:!0,get:()=>r.value,set:o=>r.value=o}):e[n]=r}}function $h(i,e,t){os(lt(i)?i.map(n=>n.bind(e.proxy)):i.bind(e.proxy),e,t)}function ig(i,e,t,n){let s=n.includes(".")?j0(t,n):()=>t[n];if(hn(i)){const r=e[i];pt(r)&&uu(s,r)}else if(pt(i))uu(s,i.bind(t));else if(Nt(i))if(lt(i))i.forEach(r=>ig(r,e,t,n));else{const r=pt(i.handler)?i.handler.bind(t):e[i.handler];pt(r)&&uu(s,r,i)}}function sg(i){const e=i.type,{mixins:t,extends:n}=e,{mixins:s,optionsCache:r,config:{optionMergeStrategies:o}}=i.appContext,a=r.get(e);let l;return a?l=a:!s.length&&!t&&!n?l=e:(l={},s.length&&s.forEach(c=>gc(l,c,o,!0)),gc(l,e,o)),Nt(e)&&r.set(e,l),l}function gc(i,e,t,n=!1){const{mixins:s,extends:r}=e;r&&gc(i,r,t,!0),s&&s.forEach(o=>gc(i,o,t,!0));for(const o in e)if(!(n&&o==="expose")){const a=Tv[o]||t&&t[o];i[o]=a?a(i[o],e[o]):e[o]}return i}const Tv={data:Zh,props:Jh,emits:Jh,methods:va,computed:va,beforeCreate:kn,created:kn,beforeMount:kn,mounted:kn,beforeUpdate:kn,updated:kn,beforeDestroy:kn,beforeUnmount:kn,destroyed:kn,unmounted:kn,activated:kn,deactivated:kn,errorCaptured:kn,serverPrefetch:kn,components:va,directives:va,watch:wv,provide:Zh,inject:Ev};function Zh(i,e){return e?i?function(){return On(pt(i)?i.call(this,this):i,pt(e)?e.call(this,this):e)}:e:i}function Ev(i,e){return va(Sf(i),Sf(e))}function Sf(i){if(lt(i)){const e={};for(let t=0;t<i.length;t++)e[i[t]]=i[t];return e}return i}function kn(i,e){return i?[...new Set([].concat(i,e))]:e}function va(i,e){return i?On(Object.create(null),i,e):e}function Jh(i,e){return i?lt(i)&&lt(e)?[...new Set([...i,...e])]:On(Object.create(null),jh(i),jh(e??{})):e}function wv(i,e){if(!i)return e;if(!e)return i;const t=On(Object.create(null),i);for(const n in e)t[n]=kn(i[n],e[n]);return t}function rg(){return{app:null,config:{isNativeTag:v0,performance:!1,globalProperties:{},optionMergeStrategies:{},errorHandler:void 0,warnHandler:void 0,compilerOptions:{}},mixins:[],components:{},directives:{},provides:Object.create(null),optionsCache:new WeakMap,propsCache:new WeakMap,emitsCache:new WeakMap}}let Rv=0;function Iv(i,e){return function(n,s=null){pt(n)||(n=On({},n)),s!=null&&!Nt(s)&&(s=null);const r=rg(),o=new WeakSet,a=[];let l=!1;const c=r.app={_uid:Rv++,_component:n,_props:s,_container:null,_context:r,_instance:null,version:uA,get config(){return r.config},set config(u){},use(u,...f){return o.has(u)||(u&&pt(u.install)?(o.add(u),u.install(c,...f)):pt(u)&&(o.add(u),u(c,...f))),c},mixin(u){return r.mixins.includes(u)||r.mixins.push(u),c},component(u,f){return f?(r.components[u]=f,c):r.components[u]},directive(u,f){return f?(r.directives[u]=f,c):r.directives[u]},mount(u,f,d){if(!l){const h=c._ceVNode||ss(n,s);return h.appContext=r,d===!0?d="svg":d===!1&&(d=void 0),i(h,u,d),l=!0,c._container=u,u.__vue_app__=c,Vc(h.component)}},onUnmount(u){a.push(u)},unmount(){l&&(os(a,c._instance,16),i(null,c._container),delete c._container.__vue_app__)},provide(u,f){return r.provides[u]=f,c},runWithContext(u){const f=Ro;Ro=c;try{return u()}finally{Ro=f}}};return c}}let Ro=null;const Dv=(i,e)=>e==="modelValue"||e==="model-value"?i.modelModifiers:i[`${e}Modifiers`]||i[`${ar(e)}Modifiers`]||i[`${hr(e)}Modifiers`];function Pv(i,e,...t){if(i.isUnmounted)return;const n=i.vnode.props||Vt;let s=t;const r=e.startsWith("update:"),o=r&&Dv(n,e.slice(7));o&&(o.trim&&(s=t.map(u=>hn(u)?u.trim():u)),o.number&&(s=t.map(Pd)));let a,l=n[a=ru(e)]||n[a=ru(ar(e))];!l&&r&&(l=n[a=ru(hr(e))]),l&&os(l,i,6,s);const c=n[a+"Once"];if(c){if(!i.emitted)i.emitted={};else if(i.emitted[a])return;i.emitted[a]=!0,os(c,i,6,s)}}const Fv=new WeakMap;function og(i,e,t=!1){const n=t?Fv:e.emitsCache,s=n.get(i);if(s!==void 0)return s;const r=i.emits;let o={},a=!1;if(!pt(i)){const l=c=>{const u=og(c,e,!0);u&&(a=!0,On(o,u))};!t&&e.mixins.length&&e.mixins.forEach(l),i.extends&&l(i.extends),i.mixins&&i.mixins.forEach(l)}return!r&&!a?(Nt(i)&&n.set(i,null),null):(lt(r)?r.forEach(l=>o[l]=null):On(o,r),Nt(i)&&n.set(i,o),o)}function kc(i,e){return!i||!Fc(e)?!1:(e=e.slice(2).replace(/Once$/,""),It(i,e[0].toLowerCase()+e.slice(1))||It(i,hr(e))||It(i,e))}function ep(i){const{type:e,vnode:t,proxy:n,withProxy:s,propsOptions:[r],slots:o,attrs:a,emit:l,render:c,renderCache:u,props:f,data:d,setupState:h,ctx:x,inheritAttrs:p}=i,g=pc(i);let m,_;try{if(t.shapeFlag&4){const A=s||n,S=A;m=ji(c.call(S,A,u,f,h,d,x)),_=a}else{const A=e;m=ji(A.length>1?A(f,{attrs:a,slots:o,emit:l}):A(f,null)),_=e.props?a:Lv(a)}}catch(A){Ra.length=0,Nc(A,i,1),m=ss(lr)}let v=m;if(_&&p!==!1){const A=Object.keys(_),{shapeFlag:S}=v;A.length&&S&7&&(r&&A.some(Rd)&&(_=Bv(_,r)),v=ko(v,_,!1,!0))}return t.dirs&&(v=ko(v,null,!1,!0),v.dirs=v.dirs?v.dirs.concat(t.dirs):t.dirs),t.transition&&Gd(v,t.transition),m=v,pc(g),m}const Lv=i=>{let e;for(const t in i)(t==="class"||t==="style"||Fc(t))&&((e||(e={}))[t]=i[t]);return e},Bv=(i,e)=>{const t={};for(const n in i)(!Rd(n)||!(n.slice(9)in e))&&(t[n]=i[n]);return t};function Uv(i,e,t){const{props:n,children:s,component:r}=i,{props:o,children:a,patchFlag:l}=e,c=r.emitsOptions;if(e.dirs||e.transition)return!0;if(t&&l>=0){if(l&1024)return!0;if(l&16)return n?tp(n,o,c):!!o;if(l&8){const u=e.dynamicProps;for(let f=0;f<u.length;f++){const d=u[f];if(ag(o,n,d)&&!kc(c,d))return!0}}}else return(s||a)&&(!a||!a.$stable)?!0:n===o?!1:n?o?tp(n,o,c):!0:!!o;return!1}function tp(i,e,t){const n=Object.keys(e);if(n.length!==Object.keys(i).length)return!0;for(let s=0;s<n.length;s++){const r=n[s];if(ag(e,i,r)&&!kc(t,r))return!0}return!1}function ag(i,e,t){const n=i[t],s=e[t];return t==="style"&&Nt(n)&&Nt(s)?!ll(n,s):n!==s}function Ov({vnode:i,parent:e},t){for(;e;){const n=e.subTree;if(n.suspense&&n.suspense.activeBranch===i&&(n.el=i.el),n===i)(i=e.vnode).el=t,e=e.parent;else break}}const lg={},cg=()=>Object.create(lg),ug=i=>Object.getPrototypeOf(i)===lg;function Nv(i,e,t,n=!1){const s={},r=cg();i.propsDefaults=Object.create(null),fg(i,e,s,r);for(const o in i.propsOptions[0])o in s||(s[o]=void 0);t?i.props=n?s:W_(s):i.type.props?i.props=s:i.props=r,i.attrs=r}function zv(i,e,t,n){const{props:s,attrs:r,vnode:{patchFlag:o}}=i,a=Rt(s),[l]=i.propsOptions;let c=!1;if((n||o>0)&&!(o&16)){if(o&8){const u=i.vnode.dynamicProps;for(let f=0;f<u.length;f++){let d=u[f];if(kc(i.emitsOptions,d))continue;const h=e[d];if(l)if(It(r,d))h!==r[d]&&(r[d]=h,c=!0);else{const x=ar(d);s[x]=yf(l,a,x,h,i,!1)}else h!==r[d]&&(r[d]=h,c=!0)}}}else{fg(i,e,s,r)&&(c=!0);let u;for(const f in a)(!e||!It(e,f)&&((u=hr(f))===f||!It(e,u)))&&(l?t&&(t[f]!==void 0||t[u]!==void 0)&&(s[f]=yf(l,a,f,void 0,i,!0)):delete s[f]);if(r!==a)for(const f in r)(!e||!It(e,f))&&(delete r[f],c=!0)}c&&Es(i.attrs,"set","")}function fg(i,e,t,n){const[s,r]=i.propsOptions;let o=!1,a;if(e)for(let l in e){if(ba(l))continue;const c=e[l];let u;s&&It(s,u=ar(l))?!r||!r.includes(u)?t[u]=c:(a||(a={}))[u]=c:kc(i.emitsOptions,l)||(!(l in n)||c!==n[l])&&(n[l]=c,o=!0)}if(r){const l=Rt(t),c=a||Vt;for(let u=0;u<r.length;u++){const f=r[u];t[f]=yf(s,l,f,c[f],i,!It(c,f))}}return o}function yf(i,e,t,n,s,r){const o=i[t];if(o!=null){const a=It(o,"default");if(a&&n===void 0){const l=o.default;if(o.type!==Function&&!o.skipFactory&&pt(l)){const{propsDefaults:c}=s;if(t in c)n=c[t];else{const u=ul(s);n=c[t]=l.call(null,e),u()}}else n=l;s.ce&&s.ce._setProp(t,n)}o[0]&&(r&&!a?n=!1:o[1]&&(n===""||n===hr(t))&&(n=!0))}return n}const kv=new WeakMap;function dg(i,e,t=!1){const n=t?kv:e.propsCache,s=n.get(i);if(s)return s;const r=i.props,o={},a=[];let l=!1;if(!pt(i)){const u=f=>{l=!0;const[d,h]=dg(f,e,!0);On(o,d),h&&a.push(...h)};!t&&e.mixins.length&&e.mixins.forEach(u),i.extends&&u(i.extends),i.mixins&&i.mixins.forEach(u)}if(!r&&!l)return Nt(i)&&n.set(i,To),To;if(lt(r))for(let u=0;u<r.length;u++){const f=ar(r[u]);np(f)&&(o[f]=Vt)}else if(r)for(const u in r){const f=ar(u);if(np(f)){const d=r[u],h=o[f]=lt(d)||pt(d)?{type:d}:On({},d),x=h.type;let p=!1,g=!0;if(lt(x))for(let m=0;m<x.length;++m){const _=x[m],v=pt(_)&&_.name;if(v==="Boolean"){p=!0;break}else v==="String"&&(g=!1)}else p=pt(x)&&x.name==="Boolean";h[0]=p,h[1]=g,(p||It(h,"default"))&&a.push(f)}}const c=[o,a];return Nt(i)&&n.set(i,c),c}function np(i){return i[0]!=="$"&&!ba(i)}const Wd=i=>i==="_"||i==="_ctx"||i==="$stable",Xd=i=>lt(i)?i.map(ji):[ji(i)],Hv=(i,e,t)=>{if(e._n)return e;const n=iv((...s)=>Xd(e(...s)),t);return n._c=!1,n},hg=(i,e,t)=>{const n=i._ctx;for(const s in i){if(Wd(s))continue;const r=i[s];if(pt(r))e[s]=Hv(s,r,n);else if(r!=null){const o=Xd(r);e[s]=()=>o}}},pg=(i,e)=>{const t=Xd(e);i.slots.default=()=>t},mg=(i,e,t)=>{for(const n in e)(t||!Wd(n))&&(i[n]=e[n])},Vv=(i,e,t)=>{const n=i.slots=cg();if(i.vnode.shapeFlag&32){const s=e._;s?(mg(n,e,t),t&&M0(n,"_",s,!0)):hg(e,n)}else e&&pg(i,e)},Gv=(i,e,t)=>{const{vnode:n,slots:s}=i;let r=!0,o=Vt;if(n.shapeFlag&32){const a=e._;a?t&&a===1?r=!1:mg(s,e,t):(r=!e.$stable,hg(e,s)),o=e}else e&&(pg(i,e),o={default:1});if(r)for(const a in s)!Wd(a)&&o[a]==null&&delete s[a]},Zn=Qv;function Wv(i){return Xv(i)}function Xv(i,e){const t=Uc();t.__VUE__=!0;const{insert:n,remove:s,patchProp:r,createElement:o,createText:a,createComment:l,setText:c,setElementText:u,parentNode:f,nextSibling:d,setScopeId:h=is,insertStaticContent:x}=i,p=(O,z,Q,w=null,ne=null,oe=null,he=void 0,ie=null,pe=!!z.dynamicChildren)=>{if(O===z)return;O&&!ua(O,z)&&(w=ue(O),Ie(O,ne,oe,!0),O=null),z.patchFlag===-2&&(pe=!1,z.dynamicChildren=null);const{type:te,ref:ve,shapeFlag:R}=z;switch(te){case Hc:g(O,z,Q,w);break;case lr:m(O,z,Q,w);break;case hu:O==null&&_(z,Q,w,he);break;case Ki:B(O,z,Q,w,ne,oe,he,ie,pe);break;default:R&1?S(O,z,Q,w,ne,oe,he,ie,pe):R&6?N(O,z,Q,w,ne,oe,he,ie,pe):(R&64||R&128)&&te.process(O,z,Q,w,ne,oe,he,ie,pe,Ce)}ve!=null&&ne?Ta(ve,O&&O.ref,oe,z||O,!z):ve==null&&O&&O.ref!=null&&Ta(O.ref,null,oe,O,!0)},g=(O,z,Q,w)=>{if(O==null)n(z.el=a(z.children),Q,w);else{const ne=z.el=O.el;z.children!==O.children&&c(ne,z.children)}},m=(O,z,Q,w)=>{O==null?n(z.el=l(z.children||""),Q,w):z.el=O.el},_=(O,z,Q,w)=>{[O.el,O.anchor]=x(O.children,z,Q,w,O.el,O.anchor)},v=({el:O,anchor:z},Q,w)=>{let ne;for(;O&&O!==z;)ne=d(O),n(O,Q,w),O=ne;n(z,Q,w)},A=({el:O,anchor:z})=>{let Q;for(;O&&O!==z;)Q=d(O),s(O),O=Q;s(z)},S=(O,z,Q,w,ne,oe,he,ie,pe)=>{if(z.type==="svg"?he="svg":z.type==="math"&&(he="mathml"),O==null)y(z,Q,w,ne,oe,he,ie,pe);else{const te=O.el&&O.el._isVueCE?O.el:null;try{te&&te._beginPatch(),b(O,z,ne,oe,he,ie,pe)}finally{te&&te._endPatch()}}},y=(O,z,Q,w,ne,oe,he,ie)=>{let pe,te;const{props:ve,shapeFlag:R,transition:T,dirs:X}=O;if(pe=O.el=o(O.type,oe,ve&&ve.is,ve),R&8?u(pe,O.children):R&16&&E(O.children,pe,null,w,ne,du(O,oe),he,ie),X&&_r(O,null,w,"created"),M(pe,O,O.scopeId,he,w),ve){for(const fe in ve)fe!=="value"&&!ba(fe)&&r(pe,fe,null,ve[fe],oe,w);"value"in ve&&r(pe,"value",null,ve.value,oe),(te=ve.onVnodeBeforeMount)&&qi(te,w,O)}X&&_r(O,null,w,"beforeMount");const se=qv(ne,T);se&&T.beforeEnter(pe),n(pe,z,Q),((te=ve&&ve.onVnodeMounted)||se||X)&&Zn(()=>{te&&qi(te,w,O),se&&T.enter(pe),X&&_r(O,null,w,"mounted")},ne)},M=(O,z,Q,w,ne)=>{if(Q&&h(O,Q),w)for(let oe=0;oe<w.length;oe++)h(O,w[oe]);if(ne){let oe=ne.subTree;if(z===oe||vg(oe.type)&&(oe.ssContent===z||oe.ssFallback===z)){const he=ne.vnode;M(O,he,he.scopeId,he.slotScopeIds,ne.parent)}}},E=(O,z,Q,w,ne,oe,he,ie,pe=0)=>{for(let te=pe;te<O.length;te++){const ve=O[te]=ie?Ms(O[te]):ji(O[te]);p(null,ve,z,Q,w,ne,oe,he,ie)}},b=(O,z,Q,w,ne,oe,he)=>{const ie=z.el=O.el;let{patchFlag:pe,dynamicChildren:te,dirs:ve}=z;pe|=O.patchFlag&16;const R=O.props||Vt,T=z.props||Vt;let X;if(Q&&vr(Q,!1),(X=T.onVnodeBeforeUpdate)&&qi(X,Q,z,O),ve&&_r(z,O,Q,"beforeUpdate"),Q&&vr(Q,!0),(R.innerHTML&&T.innerHTML==null||R.textContent&&T.textContent==null)&&u(ie,""),te?C(O.dynamicChildren,te,ie,Q,w,du(z,ne),oe):he||G(O,z,ie,null,Q,w,du(z,ne),oe,!1),pe>0){if(pe&16)P(ie,R,T,Q,ne);else if(pe&2&&R.class!==T.class&&r(ie,"class",null,T.class,ne),pe&4&&r(ie,"style",R.style,T.style,ne),pe&8){const se=z.dynamicProps;for(let fe=0;fe<se.length;fe++){const $=se[fe],Be=R[$],be=T[$];(be!==Be||$==="value")&&r(ie,$,Be,be,ne,Q)}}pe&1&&O.children!==z.children&&u(ie,z.children)}else!he&&te==null&&P(ie,R,T,Q,ne);((X=T.onVnodeUpdated)||ve)&&Zn(()=>{X&&qi(X,Q,z,O),ve&&_r(z,O,Q,"updated")},w)},C=(O,z,Q,w,ne,oe,he)=>{for(let ie=0;ie<z.length;ie++){const pe=O[ie],te=z[ie],ve=pe.el&&(pe.type===Ki||!ua(pe,te)||pe.shapeFlag&198)?f(pe.el):Q;p(pe,te,ve,null,w,ne,oe,he,!0)}},P=(O,z,Q,w,ne)=>{if(z!==Q){if(z!==Vt)for(const oe in z)!ba(oe)&&!(oe in Q)&&r(O,oe,z[oe],null,ne,w);for(const oe in Q){if(ba(oe))continue;const he=Q[oe],ie=z[oe];he!==ie&&oe!=="value"&&r(O,oe,ie,he,ne,w)}"value"in Q&&r(O,"value",z.value,Q.value,ne)}},B=(O,z,Q,w,ne,oe,he,ie,pe)=>{const te=z.el=O?O.el:a(""),ve=z.anchor=O?O.anchor:a("");let{patchFlag:R,dynamicChildren:T,slotScopeIds:X}=z;X&&(ie=ie?ie.concat(X):X),O==null?(n(te,Q,w),n(ve,Q,w),E(z.children||[],Q,ve,ne,oe,he,ie,pe)):R>0&&R&64&&T&&O.dynamicChildren&&O.dynamicChildren.length===T.length?(C(O.dynamicChildren,T,Q,ne,oe,he,ie),(z.key!=null||ne&&z===ne.subTree)&&gg(O,z,!0)):G(O,z,Q,ve,ne,oe,he,ie,pe)},N=(O,z,Q,w,ne,oe,he,ie,pe)=>{z.slotScopeIds=ie,O==null?z.shapeFlag&512?ne.ctx.activate(z,Q,w,he,pe):k(z,Q,w,ne,oe,he,pe):V(O,z,pe)},k=(O,z,Q,w,ne,oe,he)=>{const ie=O.component=iA(O,w,ne);if(Z0(O)&&(ie.ctx.renderer=Ce),rA(ie,!1,he),ie.asyncDep){if(ne&&ne.registerDep(ie,F,he),!O.el){const pe=ie.subTree=ss(lr);m(null,pe,z,Q),O.placeholder=pe.el}}else F(ie,O,z,Q,ne,oe,he)},V=(O,z,Q)=>{const w=z.component=O.component;if(Uv(O,z,Q))if(w.asyncDep&&!w.asyncResolved){q(w,z,Q);return}else w.next=z,w.update();else z.el=O.el,w.vnode=z},F=(O,z,Q,w,ne,oe,he)=>{const ie=()=>{if(O.isMounted){let{next:R,bu:T,u:X,parent:se,vnode:fe}=O;{const H=xg(O);if(H){R&&(R.el=fe.el,q(O,R,he)),H.asyncDep.then(()=>{Zn(()=>{O.isUnmounted||te()},ne)});return}}let $=R,Be;vr(O,!1),R?(R.el=fe.el,q(O,R,he)):R=fe,T&&ec(T),(Be=R.props&&R.props.onVnodeBeforeUpdate)&&qi(Be,se,R,fe),vr(O,!0);const be=ep(O),Xe=O.subTree;O.subTree=be,p(Xe,be,f(Xe.el),ue(Xe),O,ne,oe),R.el=be.el,$===null&&Ov(O,be.el),X&&Zn(X,ne),(Be=R.props&&R.props.onVnodeUpdated)&&Zn(()=>qi(Be,se,R,fe),ne)}else{let R;const{el:T,props:X}=z,{bm:se,m:fe,parent:$,root:Be,type:be}=O,Xe=Ea(z);vr(O,!1),se&&ec(se),!Xe&&(R=X&&X.onVnodeBeforeMount)&&qi(R,$,z),vr(O,!0);{Be.ce&&Be.ce._hasShadowRoot()&&Be.ce._injectChildStyle(be);const H=O.subTree=ep(O);p(null,H,Q,w,O,ne,oe),z.el=H.el}if(fe&&Zn(fe,ne),!Xe&&(R=X&&X.onVnodeMounted)){const H=z;Zn(()=>qi(R,$,H),ne)}(z.shapeFlag&256||$&&Ea($.vnode)&&$.vnode.shapeFlag&256)&&O.a&&Zn(O.a,ne),O.isMounted=!0,z=Q=w=null}};O.scope.on();const pe=O.effect=new R0(ie);O.scope.off();const te=O.update=pe.run.bind(pe),ve=O.job=pe.runIfDirty.bind(pe);ve.i=O,ve.id=O.uid,pe.scheduler=()=>Vd(ve),vr(O,!0),te()},q=(O,z,Q)=>{z.component=O;const w=O.vnode.props;O.vnode=z,O.next=null,zv(O,z.props,w,Q),Gv(O,z.children,Q),Fs(),Yh(O),Ls()},G=(O,z,Q,w,ne,oe,he,ie,pe=!1)=>{const te=O&&O.children,ve=O?O.shapeFlag:0,R=z.children,{patchFlag:T,shapeFlag:X}=z;if(T>0){if(T&128){ce(te,R,Q,w,ne,oe,he,ie,pe);return}else if(T&256){j(te,R,Q,w,ne,oe,he,ie,pe);return}}X&8?(ve&16&&ee(te,ne,oe),R!==te&&u(Q,R)):ve&16?X&16?ce(te,R,Q,w,ne,oe,he,ie,pe):ee(te,ne,oe,!0):(ve&8&&u(Q,""),X&16&&E(R,Q,w,ne,oe,he,ie,pe))},j=(O,z,Q,w,ne,oe,he,ie,pe)=>{O=O||To,z=z||To;const te=O.length,ve=z.length,R=Math.min(te,ve);let T;for(T=0;T<R;T++){const X=z[T]=pe?Ms(z[T]):ji(z[T]);p(O[T],X,Q,null,ne,oe,he,ie,pe)}te>ve?ee(O,ne,oe,!0,!1,R):E(z,Q,w,ne,oe,he,ie,pe,R)},ce=(O,z,Q,w,ne,oe,he,ie,pe)=>{let te=0;const ve=z.length;let R=O.length-1,T=ve-1;for(;te<=R&&te<=T;){const X=O[te],se=z[te]=pe?Ms(z[te]):ji(z[te]);if(ua(X,se))p(X,se,Q,null,ne,oe,he,ie,pe);else break;te++}for(;te<=R&&te<=T;){const X=O[R],se=z[T]=pe?Ms(z[T]):ji(z[T]);if(ua(X,se))p(X,se,Q,null,ne,oe,he,ie,pe);else break;R--,T--}if(te>R){if(te<=T){const X=T+1,se=X<ve?z[X].el:w;for(;te<=T;)p(null,z[te]=pe?Ms(z[te]):ji(z[te]),Q,se,ne,oe,he,ie,pe),te++}}else if(te>T)for(;te<=R;)Ie(O[te],ne,oe,!0),te++;else{const X=te,se=te,fe=new Map;for(te=se;te<=T;te++){const we=z[te]=pe?Ms(z[te]):ji(z[te]);we.key!=null&&fe.set(we.key,te)}let $,Be=0;const be=T-se+1;let Xe=!1,H=0;const J=new Array(be);for(te=0;te<be;te++)J[te]=0;for(te=X;te<=R;te++){const we=O[te];if(Be>=be){Ie(we,ne,oe,!0);continue}let Le;if(we.key!=null)Le=fe.get(we.key);else for($=se;$<=T;$++)if(J[$-se]===0&&ua(we,z[$])){Le=$;break}Le===void 0?Ie(we,ne,oe,!0):(J[Le-se]=te+1,Le>=H?H=Le:Xe=!0,p(we,z[Le],Q,null,ne,oe,he,ie,pe),Be++)}const _e=Xe?Yv(J):To;for($=_e.length-1,te=be-1;te>=0;te--){const we=se+te,Le=z[we],Re=z[we+1],$e=we+1<ve?Re.el||_g(Re):w;J[te]===0?p(null,Le,Q,$e,ne,oe,he,ie,pe):Xe&&($<0||te!==_e[$]?ye(Le,Q,$e,2):$--)}}},ye=(O,z,Q,w,ne=null)=>{const{el:oe,type:he,transition:ie,children:pe,shapeFlag:te}=O;if(te&6){ye(O.component.subTree,z,Q,w);return}if(te&128){O.suspense.move(z,Q,w);return}if(te&64){he.move(O,z,Q,Ce);return}if(he===Ki){n(oe,z,Q);for(let R=0;R<pe.length;R++)ye(pe[R],z,Q,w);n(O.anchor,z,Q);return}if(he===hu){v(O,z,Q);return}if(w!==2&&te&1&&ie)if(w===0)ie.beforeEnter(oe),n(oe,z,Q),Zn(()=>ie.enter(oe),ne);else{const{leave:R,delayLeave:T,afterLeave:X}=ie,se=()=>{O.ctx.isUnmounted?s(oe):n(oe,z,Q)},fe=()=>{oe._isLeaving&&oe[uv](!0),R(oe,()=>{se(),X&&X()})};T?T(oe,se,fe):fe()}else n(oe,z,Q)},Ie=(O,z,Q,w=!1,ne=!1)=>{const{type:oe,props:he,ref:ie,children:pe,dynamicChildren:te,shapeFlag:ve,patchFlag:R,dirs:T,cacheIndex:X}=O;if(R===-2&&(ne=!1),ie!=null&&(Fs(),Ta(ie,null,Q,O,!0),Ls()),X!=null&&(z.renderCache[X]=void 0),ve&256){z.ctx.deactivate(O);return}const se=ve&1&&T,fe=!Ea(O);let $;if(fe&&($=he&&he.onVnodeBeforeUnmount)&&qi($,z,O),ve&6)qe(O.component,Q,w);else{if(ve&128){O.suspense.unmount(Q,w);return}se&&_r(O,null,z,"beforeUnmount"),ve&64?O.type.remove(O,z,Q,Ce,w):te&&!te.hasOnce&&(oe!==Ki||R>0&&R&64)?ee(te,z,Q,!1,!0):(oe===Ki&&R&384||!ne&&ve&16)&&ee(pe,z,Q),w&&Ne(O)}(fe&&($=he&&he.onVnodeUnmounted)||se)&&Zn(()=>{$&&qi($,z,O),se&&_r(O,null,z,"unmounted")},Q)},Ne=O=>{const{type:z,el:Q,anchor:w,transition:ne}=O;if(z===Ki){de(Q,w);return}if(z===hu){A(O);return}const oe=()=>{s(Q),ne&&!ne.persisted&&ne.afterLeave&&ne.afterLeave()};if(O.shapeFlag&1&&ne&&!ne.persisted){const{leave:he,delayLeave:ie}=ne,pe=()=>he(Q,oe);ie?ie(O.el,oe,pe):pe()}else oe()},de=(O,z)=>{let Q;for(;O!==z;)Q=d(O),s(O),O=Q;s(z)},qe=(O,z,Q)=>{const{bum:w,scope:ne,job:oe,subTree:he,um:ie,m:pe,a:te}=O;ip(pe),ip(te),w&&ec(w),ne.stop(),oe&&(oe.flags|=8,Ie(he,O,z,Q)),ie&&Zn(ie,z),Zn(()=>{O.isUnmounted=!0},z)},ee=(O,z,Q,w=!1,ne=!1,oe=0)=>{for(let he=oe;he<O.length;he++)Ie(O[he],z,Q,w,ne)},ue=O=>{if(O.shapeFlag&6)return ue(O.component.subTree);if(O.shapeFlag&128)return O.suspense.next();const z=d(O.anchor||O.el),Q=z&&z[lv];return Q?d(Q):z};let Me=!1;const Ge=(O,z,Q)=>{let w;O==null?z._vnode&&(Ie(z._vnode,null,null,!0),w=z._vnode.component):p(z._vnode||null,O,z,null,null,null,Q),z._vnode=O,Me||(Me=!0,Yh(w),q0(),Me=!1)},Ce={p,um:Ie,m:ye,r:Ne,mt:k,mc:E,pc:G,pbc:C,n:ue,o:i};return{render:Ge,hydrate:void 0,createApp:Iv(Ge)}}function du({type:i,props:e},t){return t==="svg"&&i==="foreignObject"||t==="mathml"&&i==="annotation-xml"&&e&&e.encoding&&e.encoding.includes("html")?void 0:t}function vr({effect:i,job:e},t){t?(i.flags|=32,e.flags|=4):(i.flags&=-33,e.flags&=-5)}function qv(i,e){return(!i||i&&!i.pendingBranch)&&e&&!e.persisted}function gg(i,e,t=!1){const n=i.children,s=e.children;if(lt(n)&&lt(s))for(let r=0;r<n.length;r++){const o=n[r];let a=s[r];a.shapeFlag&1&&!a.dynamicChildren&&((a.patchFlag<=0||a.patchFlag===32)&&(a=s[r]=Ms(s[r]),a.el=o.el),!t&&a.patchFlag!==-2&&gg(o,a)),a.type===Hc&&(a.patchFlag===-1&&(a=s[r]=Ms(a)),a.el=o.el),a.type===lr&&!a.el&&(a.el=o.el)}}function Yv(i){const e=i.slice(),t=[0];let n,s,r,o,a;const l=i.length;for(n=0;n<l;n++){const c=i[n];if(c!==0){if(s=t[t.length-1],i[s]<c){e[n]=s,t.push(n);continue}for(r=0,o=t.length-1;r<o;)a=r+o>>1,i[t[a]]<c?r=a+1:o=a;c<i[t[r]]&&(r>0&&(e[n]=t[r-1]),t[r]=n)}}for(r=t.length,o=t[r-1];r-- >0;)t[r]=o,o=e[o];return t}function xg(i){const e=i.subTree.component;if(e)return e.asyncDep&&!e.asyncResolved?e:xg(e)}function ip(i){if(i)for(let e=0;e<i.length;e++)i[e].flags|=8}function _g(i){if(i.placeholder)return i.placeholder;const e=i.component;return e?_g(e.subTree):null}const vg=i=>i.__isSuspense;function Qv(i,e){e&&e.pendingBranch?lt(i)?e.effects.push(...i):e.effects.push(i):nv(i)}const Ki=Symbol.for("v-fgt"),Hc=Symbol.for("v-txt"),lr=Symbol.for("v-cmt"),hu=Symbol.for("v-stc"),Ra=[];let hi=null;function vn(i=!1){Ra.push(hi=i?null:[])}function Kv(){Ra.pop(),hi=Ra[Ra.length-1]||null}let Ga=1;function sp(i,e=!1){Ga+=i,i<0&&hi&&e&&(hi.hasOnce=!0)}function Ag(i){return i.dynamicChildren=Ga>0?hi||To:null,Kv(),Ga>0&&hi&&hi.push(i),i}function Cn(i,e,t,n,s,r){return Ag(Oe(i,e,t,n,s,r,!0))}function jv(i,e,t,n,s){return Ag(ss(i,e,t,n,s,!0))}function Sg(i){return i?i.__v_isVNode===!0:!1}function ua(i,e){return i.type===e.type&&i.key===e.key}const yg=({key:i})=>i??null,nc=({ref:i,ref_key:e,ref_for:t})=>(typeof i=="number"&&(i=""+i),i!=null?hn(i)||Ln(i)||pt(i)?{i:wi,r:i,k:e,f:!!t}:i:null);function Oe(i,e=null,t=null,n=0,s=null,r=i===Ki?0:1,o=!1,a=!1){const l={__v_isVNode:!0,__v_skip:!0,type:i,props:e,key:e&&yg(e),ref:e&&nc(e),scopeId:Q0,slotScopeIds:null,children:t,component:null,suspense:null,ssContent:null,ssFallback:null,dirs:null,transition:null,el:null,anchor:null,target:null,targetStart:null,targetAnchor:null,staticCount:0,shapeFlag:r,patchFlag:n,dynamicProps:s,dynamicChildren:null,appContext:null,ctx:wi};return a?(qd(l,t),r&128&&i.normalize(l)):t&&(l.shapeFlag|=hn(t)?8:16),Ga>0&&!o&&hi&&(l.patchFlag>0||r&6)&&l.patchFlag!==32&&hi.push(l),l}const ss=$v;function $v(i,e=null,t=null,n=0,s=null,r=!1){if((!i||i===Sv)&&(i=lr),Sg(i)){const a=ko(i,e,!0);return t&&qd(a,t),Ga>0&&!r&&hi&&(a.shapeFlag&6?hi[hi.indexOf(i)]=a:hi.push(a)),a.patchFlag=-2,a}if(cA(i)&&(i=i.__vccOpts),e){e=Zv(e);let{class:a,style:l}=e;a&&!hn(a)&&(e.class=Or(a)),Nt(l)&&(Hd(l)&&!lt(l)&&(l=On({},l)),e.style=Fd(l))}const o=hn(i)?1:vg(i)?128:cv(i)?64:Nt(i)?4:pt(i)?2:0;return Oe(i,e,t,n,s,o,r,!0)}function Zv(i){return i?Hd(i)||ug(i)?On({},i):i:null}function ko(i,e,t=!1,n=!1){const{props:s,ref:r,patchFlag:o,children:a,transition:l}=i,c=e?eA(s||{},e):s,u={__v_isVNode:!0,__v_skip:!0,type:i.type,props:c,key:c&&yg(c),ref:e&&e.ref?t&&r?lt(r)?r.concat(nc(e)):[r,nc(e)]:nc(e):r,scopeId:i.scopeId,slotScopeIds:i.slotScopeIds,children:a,target:i.target,targetStart:i.targetStart,targetAnchor:i.targetAnchor,staticCount:i.staticCount,shapeFlag:i.shapeFlag,patchFlag:e&&i.type!==Ki?o===-1?16:o|16:o,dynamicProps:i.dynamicProps,dynamicChildren:i.dynamicChildren,appContext:i.appContext,dirs:i.dirs,transition:l,component:i.component,suspense:i.suspense,ssContent:i.ssContent&&ko(i.ssContent),ssFallback:i.ssFallback&&ko(i.ssFallback),placeholder:i.placeholder,el:i.el,anchor:i.anchor,ctx:i.ctx,ce:i.ce};return l&&n&&Gd(u,l.clone(u)),u}function Jv(i=" ",e=0){return ss(Hc,null,i,e)}function ai(i="",e=!1){return e?(vn(),jv(lr,null,i)):ss(lr,null,i)}function ji(i){return i==null||typeof i=="boolean"?ss(lr):lt(i)?ss(Ki,null,i.slice()):Sg(i)?Ms(i):ss(Hc,null,String(i))}function Ms(i){return i.el===null&&i.patchFlag!==-1||i.memo?i:ko(i)}function qd(i,e){let t=0;const{shapeFlag:n}=i;if(e==null)e=null;else if(lt(e))t=16;else if(typeof e=="object")if(n&65){const s=e.default;s&&(s._c&&(s._d=!1),qd(i,s()),s._c&&(s._d=!0));return}else{t=32;const s=e._;!s&&!ug(e)?e._ctx=wi:s===3&&wi&&(wi.slots._===1?e._=1:(e._=2,i.patchFlag|=1024))}else pt(e)?(e={default:e,_ctx:wi},t=32):(e=String(e),n&64?(t=16,e=[Jv(e)]):t=8);i.children=e,i.shapeFlag|=t}function eA(...i){const e={};for(let t=0;t<i.length;t++){const n=i[t];for(const s in n)if(s==="class")e.class!==n.class&&(e.class=Or([e.class,n.class]));else if(s==="style")e.style=Fd([e.style,n.style]);else if(Fc(s)){const r=e[s],o=n[s];o&&r!==o&&!(lt(r)&&r.includes(o))&&(e[s]=r?[].concat(r,o):o)}else s!==""&&(e[s]=n[s])}return e}function qi(i,e,t,n=null){os(i,e,7,[t,n])}const tA=rg();let nA=0;function iA(i,e,t){const n=i.type,s=(e?e.appContext:i.appContext)||tA,r={uid:nA++,vnode:i,type:n,parent:e,appContext:s,root:null,next:null,subTree:null,effect:null,update:null,job:null,scope:new M_(!0),render:null,proxy:null,exposed:null,exposeProxy:null,withProxy:null,provides:e?e.provides:Object.create(s.provides),ids:e?e.ids:["",0,0],accessCache:null,renderCache:[],components:null,directives:null,propsOptions:dg(n,s),emitsOptions:og(n,s),emit:null,emitted:null,propsDefaults:Vt,inheritAttrs:n.inheritAttrs,ctx:Vt,data:Vt,props:Vt,attrs:Vt,slots:Vt,refs:Vt,setupState:Vt,setupContext:null,suspense:t,suspenseId:t?t.pendingId:0,asyncDep:null,asyncResolved:!1,isMounted:!1,isUnmounted:!1,isDeactivated:!1,bc:null,c:null,bm:null,m:null,bu:null,u:null,um:null,bum:null,da:null,a:null,rtg:null,rtc:null,ec:null,sp:null};return r.ctx={_:r},r.root=e?e.root:r,r.emit=Pv.bind(null,r),i.ce&&i.ce(r),r}let Xn=null;const sA=()=>Xn||wi;let xc,bf;{const i=Uc(),e=(t,n)=>{let s;return(s=i[t])||(s=i[t]=[]),s.push(n),r=>{s.length>1?s.forEach(o=>o(r)):s[0](r)}};xc=e("__VUE_INSTANCE_SETTERS__",t=>Xn=t),bf=e("__VUE_SSR_SETTERS__",t=>Wa=t)}const ul=i=>{const e=Xn;return xc(i),i.scope.on(),()=>{i.scope.off(),xc(e)}},rp=()=>{Xn&&Xn.scope.off(),xc(null)};function bg(i){return i.vnode.shapeFlag&4}let Wa=!1;function rA(i,e=!1,t=!1){e&&bf(e);const{props:n,children:s}=i.vnode,r=bg(i);Nv(i,n,r,e),Vv(i,s,t||e);const o=r?oA(i,e):void 0;return e&&bf(!1),o}function oA(i,e){const t=i.type;i.accessCache=Object.create(null),i.proxy=new Proxy(i.ctx,bv);const{setup:n}=t;if(n){Fs();const s=i.setupContext=n.length>1?lA(i):null,r=ul(i),o=cl(n,i,0,[i.props,s]),a=A0(o);if(Ls(),r(),(a||i.sp)&&!Ea(i)&&$0(i),a){if(o.then(rp,rp),e)return o.then(l=>{op(i,l)}).catch(l=>{Nc(l,i,0)});i.asyncDep=o}else op(i,o)}else Mg(i)}function op(i,e,t){pt(e)?i.type.__ssrInlineRender?i.ssrRender=e:i.render=e:Nt(e)&&(i.setupState=G0(e)),Mg(i)}function Mg(i,e,t){const n=i.type;i.render||(i.render=n.render||is);{const s=ul(i);Fs();try{Mv(i)}finally{Ls(),s()}}}const aA={get(i,e){return Pn(i,"get",""),i[e]}};function lA(i){const e=t=>{i.exposed=t||{}};return{attrs:new Proxy(i.attrs,aA),slots:i.slots,emit:i.emit,expose:e}}function Vc(i){return i.exposed?i.exposeProxy||(i.exposeProxy=new Proxy(G0(X_(i.exposed)),{get(e,t){if(t in e)return e[t];if(t in wa)return wa[t](i)},has(e,t){return t in e||t in wa}})):i.proxy}function cA(i){return pt(i)&&"__vccOpts"in i}const Ir=(i,e)=>j_(i,e,Wa),uA="3.5.28";let Mf;const ap=typeof window<"u"&&window.trustedTypes;if(ap)try{Mf=ap.createPolicy("vue",{createHTML:i=>i})}catch{}const Cg=Mf?i=>Mf.createHTML(i):i=>i,fA="http://www.w3.org/2000/svg",dA="http://www.w3.org/1998/Math/MathML",ys=typeof document<"u"?document:null,lp=ys&&ys.createElement("template"),hA={insert:(i,e,t)=>{e.insertBefore(i,t||null)},remove:i=>{const e=i.parentNode;e&&e.removeChild(i)},createElement:(i,e,t,n)=>{const s=e==="svg"?ys.createElementNS(fA,i):e==="mathml"?ys.createElementNS(dA,i):t?ys.createElement(i,{is:t}):ys.createElement(i);return i==="select"&&n&&n.multiple!=null&&s.setAttribute("multiple",n.multiple),s},createText:i=>ys.createTextNode(i),createComment:i=>ys.createComment(i),setText:(i,e)=>{i.nodeValue=e},setElementText:(i,e)=>{i.textContent=e},parentNode:i=>i.parentNode,nextSibling:i=>i.nextSibling,querySelector:i=>ys.querySelector(i),setScopeId(i,e){i.setAttribute(e,"")},insertStaticContent(i,e,t,n,s,r){const o=t?t.previousSibling:e.lastChild;if(s&&(s===r||s.nextSibling))for(;e.insertBefore(s.cloneNode(!0),t),!(s===r||!(s=s.nextSibling)););else{lp.innerHTML=Cg(n==="svg"?`<svg>${i}</svg>`:n==="mathml"?`<math>${i}</math>`:i);const a=lp.content;if(n==="svg"||n==="mathml"){const l=a.firstChild;for(;l.firstChild;)a.appendChild(l.firstChild);a.removeChild(l)}e.insertBefore(a,t)}return[o?o.nextSibling:e.firstChild,t?t.previousSibling:e.lastChild]}},pA=Symbol("_vtc");function mA(i,e,t){const n=i[pA];n&&(e=(e?[e,...n]:[...n]).join(" ")),e==null?i.removeAttribute("class"):t?i.setAttribute("class",e):i.className=e}const cp=Symbol("_vod"),gA=Symbol("_vsh"),xA=Symbol(""),_A=/(?:^|;)\s*display\s*:/;function vA(i,e,t){const n=i.style,s=hn(t);let r=!1;if(t&&!s){if(e)if(hn(e))for(const o of e.split(";")){const a=o.slice(0,o.indexOf(":")).trim();t[a]==null&&ic(n,a,"")}else for(const o in e)t[o]==null&&ic(n,o,"");for(const o in t)o==="display"&&(r=!0),ic(n,o,t[o])}else if(s){if(e!==t){const o=n[xA];o&&(t+=";"+o),n.cssText=t,r=_A.test(t)}}else e&&i.removeAttribute("style");cp in i&&(i[cp]=r?n.display:"",i[gA]&&(n.display="none"))}const up=/\s*!important$/;function ic(i,e,t){if(lt(t))t.forEach(n=>ic(i,e,n));else if(t==null&&(t=""),e.startsWith("--"))i.setProperty(e,t);else{const n=AA(i,e);up.test(t)?i.setProperty(hr(n),t.replace(up,""),"important"):i[n]=t}}const fp=["Webkit","Moz","ms"],pu={};function AA(i,e){const t=pu[e];if(t)return t;let n=ar(e);if(n!=="filter"&&n in i)return pu[e]=n;n=b0(n);for(let s=0;s<fp.length;s++){const r=fp[s]+n;if(r in i)return pu[e]=r}return e}const dp="http://www.w3.org/1999/xlink";function hp(i,e,t,n,s,r=y_(e)){n&&e.startsWith("xlink:")?t==null?i.removeAttributeNS(dp,e.slice(6,e.length)):i.setAttributeNS(dp,e,t):t==null||r&&!C0(t)?i.removeAttribute(e):i.setAttribute(e,r?"":rs(t)?String(t):t)}function pp(i,e,t,n,s){if(e==="innerHTML"||e==="textContent"){t!=null&&(i[e]=e==="innerHTML"?Cg(t):t);return}const r=i.tagName;if(e==="value"&&r!=="PROGRESS"&&!r.includes("-")){const a=r==="OPTION"?i.getAttribute("value")||"":i.value,l=t==null?i.type==="checkbox"?"on":"":String(t);(a!==l||!("_value"in i))&&(i.value=l),t==null&&i.removeAttribute(e),i._value=t;return}let o=!1;if(t===""||t==null){const a=typeof i[e];a==="boolean"?t=C0(t):t==null&&a==="string"?(t="",o=!0):a==="number"&&(t=0,o=!0)}try{i[e]=t}catch{}o&&i.removeAttribute(s||e)}function Lr(i,e,t,n){i.addEventListener(e,t,n)}function SA(i,e,t,n){i.removeEventListener(e,t,n)}const mp=Symbol("_vei");function yA(i,e,t,n,s=null){const r=i[mp]||(i[mp]={}),o=r[e];if(n&&o)o.value=n;else{const[a,l]=bA(e);if(n){const c=r[e]=TA(n,s);Lr(i,a,c,l)}else o&&(SA(i,a,o,l),r[e]=void 0)}}const gp=/(?:Once|Passive|Capture)$/;function bA(i){let e;if(gp.test(i)){e={};let n;for(;n=i.match(gp);)i=i.slice(0,i.length-n[0].length),e[n[0].toLowerCase()]=!0}return[i[2]===":"?i.slice(3):hr(i.slice(2)),e]}let mu=0;const MA=Promise.resolve(),CA=()=>mu||(MA.then(()=>mu=0),mu=Date.now());function TA(i,e){const t=n=>{if(!n._vts)n._vts=Date.now();else if(n._vts<=t.attached)return;os(EA(n,t.value),e,5,[n])};return t.value=i,t.attached=CA(),t}function EA(i,e){if(lt(e)){const t=i.stopImmediatePropagation;return i.stopImmediatePropagation=()=>{t.call(i),i._stopped=!0},e.map(n=>s=>!s._stopped&&n&&n(s))}else return e}const xp=i=>i.charCodeAt(0)===111&&i.charCodeAt(1)===110&&i.charCodeAt(2)>96&&i.charCodeAt(2)<123,wA=(i,e,t,n,s,r)=>{const o=s==="svg";e==="class"?mA(i,n,o):e==="style"?vA(i,t,n):Fc(e)?Rd(e)||yA(i,e,t,n,r):(e[0]==="."?(e=e.slice(1),!0):e[0]==="^"?(e=e.slice(1),!1):RA(i,e,n,o))?(pp(i,e,n),!i.tagName.includes("-")&&(e==="value"||e==="checked"||e==="selected")&&hp(i,e,n,o,r,e!=="value")):i._isVueCE&&(/[A-Z]/.test(e)||!hn(n))?pp(i,ar(e),n,r,e):(e==="true-value"?i._trueValue=n:e==="false-value"&&(i._falseValue=n),hp(i,e,n,o))};function RA(i,e,t,n){if(n)return!!(e==="innerHTML"||e==="textContent"||e in i&&xp(e)&&pt(t));if(e==="spellcheck"||e==="draggable"||e==="translate"||e==="autocorrect"||e==="sandbox"&&i.tagName==="IFRAME"||e==="form"||e==="list"&&i.tagName==="INPUT"||e==="type"&&i.tagName==="TEXTAREA")return!1;if(e==="width"||e==="height"){const s=i.tagName;if(s==="IMG"||s==="VIDEO"||s==="CANVAS"||s==="SOURCE")return!1}return xp(e)&&hn(t)?!1:e in i}const _c=i=>{const e=i.props["onUpdate:modelValue"]||!1;return lt(e)?t=>ec(e,t):e};function IA(i){i.target.composing=!0}function _p(i){const e=i.target;e.composing&&(e.composing=!1,e.dispatchEvent(new Event("input")))}const Io=Symbol("_assign");function vp(i,e,t){return e&&(i=i.trim()),t&&(i=Pd(i)),i}const fa={created(i,{modifiers:{lazy:e,trim:t,number:n}},s){i[Io]=_c(s);const r=n||s.props&&s.props.type==="number";Lr(i,e?"change":"input",o=>{o.target.composing||i[Io](vp(i.value,t,r))}),(t||r)&&Lr(i,"change",()=>{i.value=vp(i.value,t,r)}),e||(Lr(i,"compositionstart",IA),Lr(i,"compositionend",_p),Lr(i,"change",_p))},mounted(i,{value:e}){i.value=e??""},beforeUpdate(i,{value:e,oldValue:t,modifiers:{lazy:n,trim:s,number:r}},o){if(i[Io]=_c(o),i.composing)return;const a=(r||i.type==="number")&&!/^0\d/.test(i.value)?Pd(i.value):i.value,l=e??"";a!==l&&(document.activeElement===i&&i.type!=="range"&&(n&&e===t||s&&i.value.trim()===l)||(i.value=l))}},Ap={deep:!0,created(i,e,t){i[Io]=_c(t),Lr(i,"change",()=>{const n=i._modelValue,s=DA(i),r=i.checked,o=i[Io];if(lt(n)){const a=T0(n,s),l=a!==-1;if(r&&!l)o(n.concat(s));else if(!r&&l){const c=[...n];c.splice(a,1),o(c)}}else if(Lc(n)){const a=new Set(n);r?a.add(s):a.delete(s),o(a)}else o(Tg(i,r))})},mounted:Sp,beforeUpdate(i,e,t){i[Io]=_c(t),Sp(i,e,t)}};function Sp(i,{value:e,oldValue:t},n){i._modelValue=e;let s;if(lt(e))s=T0(e,n.props.value)>-1;else if(Lc(e))s=e.has(n.props.value);else{if(e===t)return;s=ll(e,Tg(i,!0))}i.checked!==s&&(i.checked=s)}function DA(i){return"_value"in i?i._value:i.value}function Tg(i,e){const t=e?"_trueValue":"_falseValue";return t in i?i[t]:e}const PA=["ctrl","shift","alt","meta"],FA={stop:i=>i.stopPropagation(),prevent:i=>i.preventDefault(),self:i=>i.target!==i.currentTarget,ctrl:i=>!i.ctrlKey,shift:i=>!i.shiftKey,alt:i=>!i.altKey,meta:i=>!i.metaKey,left:i=>"button"in i&&i.button!==0,middle:i=>"button"in i&&i.button!==1,right:i=>"button"in i&&i.button!==2,exact:(i,e)=>PA.some(t=>i[`${t}Key`]&&!e.includes(t))},bt=(i,e)=>{if(!i)return i;const t=i._withMods||(i._withMods={}),n=e.join(".");return t[n]||(t[n]=((s,...r)=>{for(let o=0;o<e.length;o++){const a=FA[e[o]];if(a&&a(s,e))return}return i(s,...r)}))},LA={esc:"escape",space:" ",up:"arrow-up",left:"arrow-left",right:"arrow-right",down:"arrow-down",delete:"backspace"},BA=(i,e)=>{const t=i._withKeys||(i._withKeys={}),n=e.join(".");return t[n]||(t[n]=(s=>{if(!("key"in s))return;const r=hr(s.key);if(e.some(o=>o===r||LA[o]===r))return i(s)}))},UA=On({patchProp:wA},hA);let yp;function OA(){return yp||(yp=Wv(UA))}const NA=((...i)=>{const e=OA().createApp(...i),{mount:t}=e;return e.mount=n=>{const s=kA(n);if(!s)return;const r=e._component;!pt(r)&&!r.render&&!r.template&&(r.template=s.innerHTML),s.nodeType===1&&(s.textContent="");const o=t(s,!1,zA(s));return s instanceof Element&&(s.removeAttribute("v-cloak"),s.setAttribute("data-v-app","")),o},e});function zA(i){if(i instanceof SVGElement)return"svg";if(typeof MathMLElement=="function"&&i instanceof MathMLElement)return"mathml"}function kA(i){return hn(i)?document.querySelector(i):i}const Yd="181",eo={ROTATE:0,DOLLY:1,PAN:2},to={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},HA=0,bp=1,VA=2,Eg=1,GA=2,As=3,as=0,ti=1,Ei=2,Ds=0,ir=1,Mp=2,Cp=3,Tp=4,wg=5,Br=100,WA=101,XA=102,qA=103,YA=104,QA=200,KA=201,jA=202,$A=203,Xa=204,qa=205,ZA=206,JA=207,eS=208,tS=209,nS=210,iS=211,sS=212,rS=213,oS=214,Cf=0,Tf=1,Ef=2,Ho=3,wf=4,Rf=5,If=6,Df=7,Rg=0,aS=1,lS=2,sr=0,cS=1,uS=2,fS=3,dS=4,hS=5,pS=6,mS=7,Ig=300,Vo=301,Go=302,Pf=303,Ff=304,Gc=306,Lf=1e3,Is=1001,Bf=1002,xi=1003,gS=1004,Sl=1005,Ri=1006,gu=1007,Nr=1008,ls=1009,Dg=1010,Pg=1011,Ya=1012,Qd=1013,Ii=1014,ki=1015,Qr=1016,Kd=1017,jd=1018,Qa=1020,Fg=35902,Lg=35899,Bg=1021,Ug=1022,Yn=1023,Wo=1026,Ka=1027,Og=1028,Wc=1029,$d=1030,Zd=1031,Do=1033,sc=33776,rc=33777,oc=33778,ac=33779,Uf=35840,Of=35841,Nf=35842,zf=35843,kf=36196,Hf=37492,Vf=37496,Gf=37808,Wf=37809,Xf=37810,qf=37811,Yf=37812,Qf=37813,Kf=37814,jf=37815,$f=37816,Zf=37817,Jf=37818,ed=37819,td=37820,nd=37821,id=36492,sd=36494,rd=36495,od=36283,ad=36284,ld=36285,cd=36286,xS=3200,_S=3201,vS=0,AS=1,js="",Ci="srgb",Xo="srgb-linear",vc="linear",Ut="srgb",no=7680,Ep=519,SS=512,yS=513,bS=514,Ng=515,MS=516,CS=517,TS=518,ES=519,wp=35044,wS=35048,Rp="300 es",Ji=2e3,Ac=2001;function zg(i){for(let e=i.length-1;e>=0;--e)if(i[e]>=65535)return!0;return!1}function Sc(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function RS(){const i=Sc("canvas");return i.style.display="block",i}const Ip={};function Dp(...i){const e="THREE."+i.shift();console.log(e,...i)}function ut(...i){const e="THREE."+i.shift();console.warn(e,...i)}function ln(...i){const e="THREE."+i.shift();console.error(e,...i)}function ja(...i){const e=i.join(" ");e in Ip||(Ip[e]=!0,ut(...i))}function IS(i,e,t){return new Promise(function(n,s){function r(){switch(i.clientWaitSync(e,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:s();break;case i.TIMEOUT_EXPIRED:setTimeout(r,t);break;default:n()}}setTimeout(r,t)})}class Kr{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){const n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){const n=this._listeners;if(n===void 0)return;const s=n[e];if(s!==void 0){const r=s.indexOf(t);r!==-1&&s.splice(r,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const n=t[e.type];if(n!==void 0){e.target=this;const s=n.slice(0);for(let r=0,o=s.length;r<o;r++)s[r].call(this,e);e.target=null}}}const In=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"];let Pp=1234567;const Ia=Math.PI/180,$a=180/Math.PI;function ta(){const i=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(In[i&255]+In[i>>8&255]+In[i>>16&255]+In[i>>24&255]+"-"+In[e&255]+In[e>>8&255]+"-"+In[e>>16&15|64]+In[e>>24&255]+"-"+In[t&63|128]+In[t>>8&255]+"-"+In[t>>16&255]+In[t>>24&255]+In[n&255]+In[n>>8&255]+In[n>>16&255]+In[n>>24&255]).toLowerCase()}function ht(i,e,t){return Math.max(e,Math.min(t,i))}function Jd(i,e){return(i%e+e)%e}function DS(i,e,t,n,s){return n+(i-e)*(s-n)/(t-e)}function PS(i,e,t){return i!==e?(t-i)/(e-i):0}function Da(i,e,t){return(1-t)*i+t*e}function FS(i,e,t,n){return Da(i,e,1-Math.exp(-t*n))}function LS(i,e=1){return e-Math.abs(Jd(i,e*2)-e)}function BS(i,e,t){return i<=e?0:i>=t?1:(i=(i-e)/(t-e),i*i*(3-2*i))}function US(i,e,t){return i<=e?0:i>=t?1:(i=(i-e)/(t-e),i*i*i*(i*(i*6-15)+10))}function OS(i,e){return i+Math.floor(Math.random()*(e-i+1))}function NS(i,e){return i+Math.random()*(e-i)}function zS(i){return i*(.5-Math.random())}function kS(i){i!==void 0&&(Pp=i);let e=Pp+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=e+Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}function HS(i){return i*Ia}function VS(i){return i*$a}function GS(i){return(i&i-1)===0&&i!==0}function WS(i){return Math.pow(2,Math.ceil(Math.log(i)/Math.LN2))}function XS(i){return Math.pow(2,Math.floor(Math.log(i)/Math.LN2))}function qS(i,e,t,n,s){const r=Math.cos,o=Math.sin,a=r(t/2),l=o(t/2),c=r((e+n)/2),u=o((e+n)/2),f=r((e-n)/2),d=o((e-n)/2),h=r((n-e)/2),x=o((n-e)/2);switch(s){case"XYX":i.set(a*u,l*f,l*d,a*c);break;case"YZY":i.set(l*d,a*u,l*f,a*c);break;case"ZXZ":i.set(l*f,l*d,a*u,a*c);break;case"XZX":i.set(a*u,l*x,l*h,a*c);break;case"YXY":i.set(l*h,a*u,l*x,a*c);break;case"ZYZ":i.set(l*x,l*h,a*u,a*c);break;default:ut("MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+s)}}function So(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("Invalid component type.")}}function Hn(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("Invalid component type.")}}const An={DEG2RAD:Ia,RAD2DEG:$a,generateUUID:ta,clamp:ht,euclideanModulo:Jd,mapLinear:DS,inverseLerp:PS,lerp:Da,damp:FS,pingpong:LS,smoothstep:BS,smootherstep:US,randInt:OS,randFloat:NS,randFloatSpread:zS,seededRandom:kS,degToRad:HS,radToDeg:VS,isPowerOfTwo:GS,ceilPowerOfTwo:WS,floorPowerOfTwo:XS,setQuaternionFromProperEuler:qS,normalize:Hn,denormalize:So};class Ke{constructor(e=0,t=0){Ke.prototype.isVector2=!0,this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,n=this.y,s=e.elements;return this.x=s[0]*t+s[3]*n+s[6],this.y=s[1]*t+s[4]*n+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=ht(this.x,e.x,t.x),this.y=ht(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=ht(this.x,e,t),this.y=ht(this.y,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(ht(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(ht(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const n=Math.cos(t),s=Math.sin(t),r=this.x-e.x,o=this.y-e.y;return this.x=r*n-o*s+e.x,this.y=r*s+o*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class Gt{constructor(e=0,t=0,n=0,s=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=s}static slerpFlat(e,t,n,s,r,o,a){let l=n[s+0],c=n[s+1],u=n[s+2],f=n[s+3],d=r[o+0],h=r[o+1],x=r[o+2],p=r[o+3];if(a<=0){e[t+0]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f;return}if(a>=1){e[t+0]=d,e[t+1]=h,e[t+2]=x,e[t+3]=p;return}if(f!==p||l!==d||c!==h||u!==x){let g=l*d+c*h+u*x+f*p;g<0&&(d=-d,h=-h,x=-x,p=-p,g=-g);let m=1-a;if(g<.9995){const _=Math.acos(g),v=Math.sin(_);m=Math.sin(m*_)/v,a=Math.sin(a*_)/v,l=l*m+d*a,c=c*m+h*a,u=u*m+x*a,f=f*m+p*a}else{l=l*m+d*a,c=c*m+h*a,u=u*m+x*a,f=f*m+p*a;const _=1/Math.sqrt(l*l+c*c+u*u+f*f);l*=_,c*=_,u*=_,f*=_}}e[t]=l,e[t+1]=c,e[t+2]=u,e[t+3]=f}static multiplyQuaternionsFlat(e,t,n,s,r,o){const a=n[s],l=n[s+1],c=n[s+2],u=n[s+3],f=r[o],d=r[o+1],h=r[o+2],x=r[o+3];return e[t]=a*x+u*f+l*h-c*d,e[t+1]=l*x+u*d+c*f-a*h,e[t+2]=c*x+u*h+a*d-l*f,e[t+3]=u*x-a*f-l*d-c*h,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,s){return this._x=e,this._y=t,this._z=n,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const n=e._x,s=e._y,r=e._z,o=e._order,a=Math.cos,l=Math.sin,c=a(n/2),u=a(s/2),f=a(r/2),d=l(n/2),h=l(s/2),x=l(r/2);switch(o){case"XYZ":this._x=d*u*f+c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f-d*h*x;break;case"YXZ":this._x=d*u*f+c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f+d*h*x;break;case"ZXY":this._x=d*u*f-c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f-d*h*x;break;case"ZYX":this._x=d*u*f-c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f+d*h*x;break;case"YZX":this._x=d*u*f+c*h*x,this._y=c*h*f+d*u*x,this._z=c*u*x-d*h*f,this._w=c*u*f-d*h*x;break;case"XZY":this._x=d*u*f-c*h*x,this._y=c*h*f-d*u*x,this._z=c*u*x+d*h*f,this._w=c*u*f+d*h*x;break;default:ut("Quaternion: .setFromEuler() encountered an unknown order: "+o)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const n=t/2,s=Math.sin(n);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,n=t[0],s=t[4],r=t[8],o=t[1],a=t[5],l=t[9],c=t[2],u=t[6],f=t[10],d=n+a+f;if(d>0){const h=.5/Math.sqrt(d+1);this._w=.25/h,this._x=(u-l)*h,this._y=(r-c)*h,this._z=(o-s)*h}else if(n>a&&n>f){const h=2*Math.sqrt(1+n-a-f);this._w=(u-l)/h,this._x=.25*h,this._y=(s+o)/h,this._z=(r+c)/h}else if(a>f){const h=2*Math.sqrt(1+a-n-f);this._w=(r-c)/h,this._x=(s+o)/h,this._y=.25*h,this._z=(l+u)/h}else{const h=2*Math.sqrt(1+f-n-a);this._w=(o-s)/h,this._x=(r+c)/h,this._y=(l+u)/h,this._z=.25*h}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(ht(this.dot(e),-1,1)))}rotateTowards(e,t){const n=this.angleTo(e);if(n===0)return this;const s=Math.min(1,t/n);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const n=e._x,s=e._y,r=e._z,o=e._w,a=t._x,l=t._y,c=t._z,u=t._w;return this._x=n*u+o*a+s*c-r*l,this._y=s*u+o*l+r*a-n*c,this._z=r*u+o*c+n*l-s*a,this._w=o*u-n*a-s*l-r*c,this._onChangeCallback(),this}slerp(e,t){if(t<=0)return this;if(t>=1)return this.copy(e);let n=e._x,s=e._y,r=e._z,o=e._w,a=this.dot(e);a<0&&(n=-n,s=-s,r=-r,o=-o,a=-a);let l=1-t;if(a<.9995){const c=Math.acos(a),u=Math.sin(c);l=Math.sin(l*c)/u,t=Math.sin(t*c)/u,this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this._onChangeCallback()}else this._x=this._x*l+n*t,this._y=this._y*l+s*t,this._z=this._z*l+r*t,this._w=this._w*l+o*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),s=Math.sqrt(1-n),r=Math.sqrt(n);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(t),r*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class U{constructor(e=0,t=0,n=0){U.prototype.isVector3=!0,this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(Fp.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(Fp.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[3]*n+r[6]*s,this.y=r[1]*t+r[4]*n+r[7]*s,this.z=r[2]*t+r[5]*n+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=e.elements,o=1/(r[3]*t+r[7]*n+r[11]*s+r[15]);return this.x=(r[0]*t+r[4]*n+r[8]*s+r[12])*o,this.y=(r[1]*t+r[5]*n+r[9]*s+r[13])*o,this.z=(r[2]*t+r[6]*n+r[10]*s+r[14])*o,this}applyQuaternion(e){const t=this.x,n=this.y,s=this.z,r=e.x,o=e.y,a=e.z,l=e.w,c=2*(o*s-a*n),u=2*(a*t-r*s),f=2*(r*n-o*t);return this.x=t+l*c+o*f-a*u,this.y=n+l*u+a*c-r*f,this.z=s+l*f+r*u-o*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,n=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[4]*n+r[8]*s,this.y=r[1]*t+r[5]*n+r[9]*s,this.z=r[2]*t+r[6]*n+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=ht(this.x,e.x,t.x),this.y=ht(this.y,e.y,t.y),this.z=ht(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=ht(this.x,e,t),this.y=ht(this.y,e,t),this.z=ht(this.z,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(ht(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const n=e.x,s=e.y,r=e.z,o=t.x,a=t.y,l=t.z;return this.x=s*l-r*a,this.y=r*o-n*l,this.z=n*a-s*o,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return xu.copy(this).projectOnVector(e),this.sub(xu)}reflect(e){return this.sub(xu.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(ht(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y,s=this.z-e.z;return t*t+n*n+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){const s=Math.sin(t)*e;return this.x=s*Math.sin(n),this.y=Math.cos(t)*e,this.z=s*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=s,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const xu=new U,Fp=new Gt;class at{constructor(e,t,n,s,r,o,a,l,c){at.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c)}set(e,t,n,s,r,o,a,l,c){const u=this.elements;return u[0]=e,u[1]=s,u[2]=a,u[3]=t,u[4]=r,u[5]=l,u[6]=n,u[7]=o,u[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[3],l=n[6],c=n[1],u=n[4],f=n[7],d=n[2],h=n[5],x=n[8],p=s[0],g=s[3],m=s[6],_=s[1],v=s[4],A=s[7],S=s[2],y=s[5],M=s[8];return r[0]=o*p+a*_+l*S,r[3]=o*g+a*v+l*y,r[6]=o*m+a*A+l*M,r[1]=c*p+u*_+f*S,r[4]=c*g+u*v+f*y,r[7]=c*m+u*A+f*M,r[2]=d*p+h*_+x*S,r[5]=d*g+h*v+x*y,r[8]=d*m+h*A+x*M,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8];return t*o*u-t*a*c-n*r*u+n*a*l+s*r*c-s*o*l}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=u*o-a*c,d=a*l-u*r,h=c*r-o*l,x=t*f+n*d+s*h;if(x===0)return this.set(0,0,0,0,0,0,0,0,0);const p=1/x;return e[0]=f*p,e[1]=(s*c-u*n)*p,e[2]=(a*n-s*o)*p,e[3]=d*p,e[4]=(u*t-s*l)*p,e[5]=(s*r-a*t)*p,e[6]=h*p,e[7]=(n*l-c*t)*p,e[8]=(o*t-n*r)*p,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,s,r,o,a){const l=Math.cos(r),c=Math.sin(r);return this.set(n*l,n*c,-n*(l*o+c*a)+o+e,-s*c,s*l,-s*(-c*o+l*a)+a+t,0,0,1),this}scale(e,t){return this.premultiply(_u.makeScale(e,t)),this}rotate(e){return this.premultiply(_u.makeRotation(-e)),this}translate(e,t){return this.premultiply(_u.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<9;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const _u=new at,Lp=new at().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Bp=new at().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function YS(){const i={enabled:!0,workingColorSpace:Xo,spaces:{},convert:function(s,r,o){return this.enabled===!1||r===o||!r||!o||(this.spaces[r].transfer===Ut&&(s.r=Ps(s.r),s.g=Ps(s.g),s.b=Ps(s.b)),this.spaces[r].primaries!==this.spaces[o].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[o].fromXYZ)),this.spaces[o].transfer===Ut&&(s.r=Po(s.r),s.g=Po(s.g),s.b=Po(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===js?vc:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,o){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[o].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return ja("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),i.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return ja("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),i.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[Xo]:{primaries:e,whitePoint:n,transfer:vc,toXYZ:Lp,fromXYZ:Bp,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:Ci},outputColorSpaceConfig:{drawingBufferColorSpace:Ci}},[Ci]:{primaries:e,whitePoint:n,transfer:Ut,toXYZ:Lp,fromXYZ:Bp,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:Ci}}}),i}const yt=YS();function Ps(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function Po(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}let io;class QS{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{io===void 0&&(io=Sc("canvas")),io.width=e.width,io.height=e.height;const s=io.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),n=io}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=Sc("canvas");t.width=e.width,t.height=e.height;const n=t.getContext("2d");n.drawImage(e,0,0,e.width,e.height);const s=n.getImageData(0,0,e.width,e.height),r=s.data;for(let o=0;o<r.length;o++)r[o]=Ps(r[o]/255)*255;return n.putImageData(s,0,0),t}else if(e.data){const t=e.data.slice(0);for(let n=0;n<t.length;n++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[n]=Math.floor(Ps(t[n]/255)*255):t[n]=Ps(t[n]);return{data:t,width:e.width,height:e.height}}else return ut("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let KS=0;class eh{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:KS++}),this.uuid=ta(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):t instanceof VideoFrame?e.set(t.displayHeight,t.displayWidth,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const n={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let o=0,a=s.length;o<a;o++)s[o].isDataTexture?r.push(vu(s[o].image)):r.push(vu(s[o]))}else r=vu(s);n.url=r}return t||(e.images[this.uuid]=n),n}}function vu(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?QS.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(ut("Texture: Unable to serialize Texture."),{})}let jS=0;const Au=new U;class Qn extends Kr{constructor(e=Qn.DEFAULT_IMAGE,t=Qn.DEFAULT_MAPPING,n=Is,s=Is,r=Ri,o=Nr,a=Yn,l=ls,c=Qn.DEFAULT_ANISOTROPY,u=js){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:jS++}),this.uuid=ta(),this.name="",this.source=new eh(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=n,this.wrapT=s,this.magFilter=r,this.minFilter=o,this.anisotropy=c,this.format=a,this.internalFormat=null,this.type=l,this.offset=new Ke(0,0),this.repeat=new Ke(1,1),this.center=new Ke(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new at,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=u,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(Au).x}get height(){return this.source.getSize(Au).y}get depth(){return this.source.getSize(Au).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const n=e[t];if(n===void 0){ut(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){ut(`Texture.setValues(): property '${t}' does not exist.`);continue}s&&n&&s.isVector2&&n.isVector2||s&&n&&s.isVector3&&n.isVector3||s&&n&&s.isMatrix3&&n.isMatrix3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==Ig)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case Lf:e.x=e.x-Math.floor(e.x);break;case Is:e.x=e.x<0?0:1;break;case Bf:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case Lf:e.y=e.y-Math.floor(e.y);break;case Is:e.y=e.y<0?0:1;break;case Bf:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}Qn.DEFAULT_IMAGE=null;Qn.DEFAULT_MAPPING=Ig;Qn.DEFAULT_ANISOTROPY=1;class Zt{constructor(e=0,t=0,n=0,s=1){Zt.prototype.isVector4=!0,this.x=e,this.y=t,this.z=n,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,s){return this.x=e,this.y=t,this.z=n,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,n=this.y,s=this.z,r=this.w,o=e.elements;return this.x=o[0]*t+o[4]*n+o[8]*s+o[12]*r,this.y=o[1]*t+o[5]*n+o[9]*s+o[13]*r,this.z=o[2]*t+o[6]*n+o[10]*s+o[14]*r,this.w=o[3]*t+o[7]*n+o[11]*s+o[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,s,r;const l=e.elements,c=l[0],u=l[4],f=l[8],d=l[1],h=l[5],x=l[9],p=l[2],g=l[6],m=l[10];if(Math.abs(u-d)<.01&&Math.abs(f-p)<.01&&Math.abs(x-g)<.01){if(Math.abs(u+d)<.1&&Math.abs(f+p)<.1&&Math.abs(x+g)<.1&&Math.abs(c+h+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const v=(c+1)/2,A=(h+1)/2,S=(m+1)/2,y=(u+d)/4,M=(f+p)/4,E=(x+g)/4;return v>A&&v>S?v<.01?(n=0,s=.707106781,r=.707106781):(n=Math.sqrt(v),s=y/n,r=M/n):A>S?A<.01?(n=.707106781,s=0,r=.707106781):(s=Math.sqrt(A),n=y/s,r=E/s):S<.01?(n=.707106781,s=.707106781,r=0):(r=Math.sqrt(S),n=M/r,s=E/r),this.set(n,s,r,t),this}let _=Math.sqrt((g-x)*(g-x)+(f-p)*(f-p)+(d-u)*(d-u));return Math.abs(_)<.001&&(_=1),this.x=(g-x)/_,this.y=(f-p)/_,this.z=(d-u)/_,this.w=Math.acos((c+h+m-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=ht(this.x,e.x,t.x),this.y=ht(this.y,e.y,t.y),this.z=ht(this.z,e.z,t.z),this.w=ht(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=ht(this.x,e,t),this.y=ht(this.y,e,t),this.z=ht(this.z,e,t),this.w=ht(this.w,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(ht(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class $S extends Kr{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:Ri,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new Zt(0,0,e,t),this.scissorTest=!1,this.viewport=new Zt(0,0,e,t);const s={width:e,height:t,depth:n.depth},r=new Qn(s);this.textures=[];const o=n.count;for(let a=0;a<o;a++)this.textures[a]=r.clone(),this.textures[a].isRenderTargetTexture=!0,this.textures[a].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview}_setTextureOptions(e={}){const t={minFilter:Ri,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=t,this.textures[s].image.depth=n,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const s=Object.assign({},e.textures[t].image);this.textures[t].source=new eh(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class cr extends $S{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}}class kg extends Qn{constructor(e=null,t=1,n=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=xi,this.minFilter=xi,this.wrapR=Is,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class ZS extends Qn{constructor(e=null,t=1,n=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:s},this.magFilter=xi,this.minFilter=xi,this.wrapR=Is,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class es{constructor(e=new U(1/0,1/0,1/0),t=new U(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(Ui.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(Ui.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const n=Ui.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const n=e.geometry;if(n!==void 0){const r=n.getAttribute("position");if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let o=0,a=r.count;o<a;o++)e.isMesh===!0?e.getVertexPosition(o,Ui):Ui.fromBufferAttribute(r,o),Ui.applyMatrix4(e.matrixWorld),this.expandByPoint(Ui);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),yl.copy(e.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),yl.copy(n.boundingBox)),yl.applyMatrix4(e.matrixWorld),this.union(yl)}const s=e.children;for(let r=0,o=s.length;r<o;r++)this.expandByObject(s[r],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,Ui),Ui.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(da),bl.subVectors(this.max,da),so.subVectors(e.a,da),ro.subVectors(e.b,da),oo.subVectors(e.c,da),ks.subVectors(ro,so),Hs.subVectors(oo,ro),Ar.subVectors(so,oo);let t=[0,-ks.z,ks.y,0,-Hs.z,Hs.y,0,-Ar.z,Ar.y,ks.z,0,-ks.x,Hs.z,0,-Hs.x,Ar.z,0,-Ar.x,-ks.y,ks.x,0,-Hs.y,Hs.x,0,-Ar.y,Ar.x,0];return!Su(t,so,ro,oo,bl)||(t=[1,0,0,0,1,0,0,0,1],!Su(t,so,ro,oo,bl))?!1:(Ml.crossVectors(ks,Hs),t=[Ml.x,Ml.y,Ml.z],Su(t,so,ro,oo,bl))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,Ui).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(Ui).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(hs[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),hs[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),hs[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),hs[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),hs[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),hs[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),hs[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),hs[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(hs),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const hs=[new U,new U,new U,new U,new U,new U,new U,new U],Ui=new U,yl=new es,so=new U,ro=new U,oo=new U,ks=new U,Hs=new U,Ar=new U,da=new U,bl=new U,Ml=new U,Sr=new U;function Su(i,e,t,n,s){for(let r=0,o=i.length-3;r<=o;r+=3){Sr.fromArray(i,r);const a=s.x*Math.abs(Sr.x)+s.y*Math.abs(Sr.y)+s.z*Math.abs(Sr.z),l=e.dot(Sr),c=t.dot(Sr),u=n.dot(Sr);if(Math.max(-Math.max(l,c,u),Math.min(l,c,u))>a)return!1}return!0}const JS=new es,ha=new U,yu=new U;class Xc{constructor(e=new U,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const n=this.center;t!==void 0?n.copy(t):JS.setFromPoints(e).getCenter(n);let s=0;for(let r=0,o=e.length;r<o;r++)s=Math.max(s,n.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;ha.subVectors(e,this.center);const t=ha.lengthSq();if(t>this.radius*this.radius){const n=Math.sqrt(t),s=(n-this.radius)*.5;this.center.addScaledVector(ha,s/n),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(yu.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(ha.copy(e.center).add(yu)),this.expandByPoint(ha.copy(e.center).sub(yu))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}const ps=new U,bu=new U,Cl=new U,Vs=new U,Mu=new U,Tl=new U,Cu=new U;let th=class{constructor(e=new U,t=new U(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,ps)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=ps.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(ps.copy(this.origin).addScaledVector(this.direction,t),ps.distanceToSquared(e))}distanceSqToSegment(e,t,n,s){bu.copy(e).add(t).multiplyScalar(.5),Cl.copy(t).sub(e).normalize(),Vs.copy(this.origin).sub(bu);const r=e.distanceTo(t)*.5,o=-this.direction.dot(Cl),a=Vs.dot(this.direction),l=-Vs.dot(Cl),c=Vs.lengthSq(),u=Math.abs(1-o*o);let f,d,h,x;if(u>0)if(f=o*l-a,d=o*a-l,x=r*u,f>=0)if(d>=-x)if(d<=x){const p=1/u;f*=p,d*=p,h=f*(f+o*d+2*a)+d*(o*f+d+2*l)+c}else d=r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;else d=-r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;else d<=-x?(f=Math.max(0,-(-o*r+a)),d=f>0?-r:Math.min(Math.max(-r,-l),r),h=-f*f+d*(d+2*l)+c):d<=x?(f=0,d=Math.min(Math.max(-r,-l),r),h=d*(d+2*l)+c):(f=Math.max(0,-(o*r+a)),d=f>0?r:Math.min(Math.max(-r,-l),r),h=-f*f+d*(d+2*l)+c);else d=o>0?-r:r,f=Math.max(0,-(o*d+a)),h=-f*f+d*(d+2*l)+c;return n&&n.copy(this.origin).addScaledVector(this.direction,f),s&&s.copy(bu).addScaledVector(Cl,d),h}intersectSphere(e,t){ps.subVectors(e.center,this.origin);const n=ps.dot(this.direction),s=ps.dot(ps)-n*n,r=e.radius*e.radius;if(s>r)return null;const o=Math.sqrt(r-s),a=n-o,l=n+o;return l<0?null:a<0?this.at(l,t):this.at(a,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){const n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,s,r,o,a,l;const c=1/this.direction.x,u=1/this.direction.y,f=1/this.direction.z,d=this.origin;return c>=0?(n=(e.min.x-d.x)*c,s=(e.max.x-d.x)*c):(n=(e.max.x-d.x)*c,s=(e.min.x-d.x)*c),u>=0?(r=(e.min.y-d.y)*u,o=(e.max.y-d.y)*u):(r=(e.max.y-d.y)*u,o=(e.min.y-d.y)*u),n>o||r>s||((r>n||isNaN(n))&&(n=r),(o<s||isNaN(s))&&(s=o),f>=0?(a=(e.min.z-d.z)*f,l=(e.max.z-d.z)*f):(a=(e.max.z-d.z)*f,l=(e.min.z-d.z)*f),n>l||a>s)||((a>n||n!==n)&&(n=a),(l<s||s!==s)&&(s=l),s<0)?null:this.at(n>=0?n:s,t)}intersectsBox(e){return this.intersectBox(e,ps)!==null}intersectTriangle(e,t,n,s,r){Mu.subVectors(t,e),Tl.subVectors(n,e),Cu.crossVectors(Mu,Tl);let o=this.direction.dot(Cu),a;if(o>0){if(s)return null;a=1}else if(o<0)a=-1,o=-o;else return null;Vs.subVectors(this.origin,e);const l=a*this.direction.dot(Tl.crossVectors(Vs,Tl));if(l<0)return null;const c=a*this.direction.dot(Mu.cross(Vs));if(c<0||l+c>o)return null;const u=-a*Vs.dot(Cu);return u<0?null:this.at(u/o,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}};class rt{constructor(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g){rt.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g)}set(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g){const m=this.elements;return m[0]=e,m[4]=t,m[8]=n,m[12]=s,m[1]=r,m[5]=o,m[9]=a,m[13]=l,m[2]=c,m[6]=u,m[10]=f,m[14]=d,m[3]=h,m[7]=x,m[11]=p,m[15]=g,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new rt().fromArray(this.elements)}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){const t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){const t=this.elements,n=e.elements,s=1/ao.setFromMatrixColumn(e,0).length(),r=1/ao.setFromMatrixColumn(e,1).length(),o=1/ao.setFromMatrixColumn(e,2).length();return t[0]=n[0]*s,t[1]=n[1]*s,t[2]=n[2]*s,t[3]=0,t[4]=n[4]*r,t[5]=n[5]*r,t[6]=n[6]*r,t[7]=0,t[8]=n[8]*o,t[9]=n[9]*o,t[10]=n[10]*o,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,n=e.x,s=e.y,r=e.z,o=Math.cos(n),a=Math.sin(n),l=Math.cos(s),c=Math.sin(s),u=Math.cos(r),f=Math.sin(r);if(e.order==="XYZ"){const d=o*u,h=o*f,x=a*u,p=a*f;t[0]=l*u,t[4]=-l*f,t[8]=c,t[1]=h+x*c,t[5]=d-p*c,t[9]=-a*l,t[2]=p-d*c,t[6]=x+h*c,t[10]=o*l}else if(e.order==="YXZ"){const d=l*u,h=l*f,x=c*u,p=c*f;t[0]=d+p*a,t[4]=x*a-h,t[8]=o*c,t[1]=o*f,t[5]=o*u,t[9]=-a,t[2]=h*a-x,t[6]=p+d*a,t[10]=o*l}else if(e.order==="ZXY"){const d=l*u,h=l*f,x=c*u,p=c*f;t[0]=d-p*a,t[4]=-o*f,t[8]=x+h*a,t[1]=h+x*a,t[5]=o*u,t[9]=p-d*a,t[2]=-o*c,t[6]=a,t[10]=o*l}else if(e.order==="ZYX"){const d=o*u,h=o*f,x=a*u,p=a*f;t[0]=l*u,t[4]=x*c-h,t[8]=d*c+p,t[1]=l*f,t[5]=p*c+d,t[9]=h*c-x,t[2]=-c,t[6]=a*l,t[10]=o*l}else if(e.order==="YZX"){const d=o*l,h=o*c,x=a*l,p=a*c;t[0]=l*u,t[4]=p-d*f,t[8]=x*f+h,t[1]=f,t[5]=o*u,t[9]=-a*u,t[2]=-c*u,t[6]=h*f+x,t[10]=d-p*f}else if(e.order==="XZY"){const d=o*l,h=o*c,x=a*l,p=a*c;t[0]=l*u,t[4]=-f,t[8]=c*u,t[1]=d*f+p,t[5]=o*u,t[9]=h*f-x,t[2]=x*f-h,t[6]=a*u,t[10]=p*f+d}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(ey,e,ty)}lookAt(e,t,n){const s=this.elements;return li.subVectors(e,t),li.lengthSq()===0&&(li.z=1),li.normalize(),Gs.crossVectors(n,li),Gs.lengthSq()===0&&(Math.abs(n.z)===1?li.x+=1e-4:li.z+=1e-4,li.normalize(),Gs.crossVectors(n,li)),Gs.normalize(),El.crossVectors(li,Gs),s[0]=Gs.x,s[4]=El.x,s[8]=li.x,s[1]=Gs.y,s[5]=El.y,s[9]=li.y,s[2]=Gs.z,s[6]=El.z,s[10]=li.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,s=t.elements,r=this.elements,o=n[0],a=n[4],l=n[8],c=n[12],u=n[1],f=n[5],d=n[9],h=n[13],x=n[2],p=n[6],g=n[10],m=n[14],_=n[3],v=n[7],A=n[11],S=n[15],y=s[0],M=s[4],E=s[8],b=s[12],C=s[1],P=s[5],B=s[9],N=s[13],k=s[2],V=s[6],F=s[10],q=s[14],G=s[3],j=s[7],ce=s[11],ye=s[15];return r[0]=o*y+a*C+l*k+c*G,r[4]=o*M+a*P+l*V+c*j,r[8]=o*E+a*B+l*F+c*ce,r[12]=o*b+a*N+l*q+c*ye,r[1]=u*y+f*C+d*k+h*G,r[5]=u*M+f*P+d*V+h*j,r[9]=u*E+f*B+d*F+h*ce,r[13]=u*b+f*N+d*q+h*ye,r[2]=x*y+p*C+g*k+m*G,r[6]=x*M+p*P+g*V+m*j,r[10]=x*E+p*B+g*F+m*ce,r[14]=x*b+p*N+g*q+m*ye,r[3]=_*y+v*C+A*k+S*G,r[7]=_*M+v*P+A*V+S*j,r[11]=_*E+v*B+A*F+S*ce,r[15]=_*b+v*N+A*q+S*ye,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[4],s=e[8],r=e[12],o=e[1],a=e[5],l=e[9],c=e[13],u=e[2],f=e[6],d=e[10],h=e[14],x=e[3],p=e[7],g=e[11],m=e[15];return x*(+r*l*f-s*c*f-r*a*d+n*c*d+s*a*h-n*l*h)+p*(+t*l*h-t*c*d+r*o*d-s*o*h+s*c*u-r*l*u)+g*(+t*c*f-t*a*h-r*o*f+n*o*h+r*a*u-n*c*u)+m*(-s*a*u-t*l*f+t*a*d+s*o*f-n*o*d+n*l*u)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){const s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=t,s[14]=n),this}invert(){const e=this.elements,t=e[0],n=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],f=e[9],d=e[10],h=e[11],x=e[12],p=e[13],g=e[14],m=e[15],_=f*g*c-p*d*c+p*l*h-a*g*h-f*l*m+a*d*m,v=x*d*c-u*g*c-x*l*h+o*g*h+u*l*m-o*d*m,A=u*p*c-x*f*c+x*a*h-o*p*h-u*a*m+o*f*m,S=x*f*l-u*p*l-x*a*d+o*p*d+u*a*g-o*f*g,y=t*_+n*v+s*A+r*S;if(y===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const M=1/y;return e[0]=_*M,e[1]=(p*d*r-f*g*r-p*s*h+n*g*h+f*s*m-n*d*m)*M,e[2]=(a*g*r-p*l*r+p*s*c-n*g*c-a*s*m+n*l*m)*M,e[3]=(f*l*r-a*d*r-f*s*c+n*d*c+a*s*h-n*l*h)*M,e[4]=v*M,e[5]=(u*g*r-x*d*r+x*s*h-t*g*h-u*s*m+t*d*m)*M,e[6]=(x*l*r-o*g*r-x*s*c+t*g*c+o*s*m-t*l*m)*M,e[7]=(o*d*r-u*l*r+u*s*c-t*d*c-o*s*h+t*l*h)*M,e[8]=A*M,e[9]=(x*f*r-u*p*r-x*n*h+t*p*h+u*n*m-t*f*m)*M,e[10]=(o*p*r-x*a*r+x*n*c-t*p*c-o*n*m+t*a*m)*M,e[11]=(u*a*r-o*f*r-u*n*c+t*f*c+o*n*h-t*a*h)*M,e[12]=S*M,e[13]=(u*p*s-x*f*s+x*n*d-t*p*d-u*n*g+t*f*g)*M,e[14]=(x*a*s-o*p*s-x*n*l+t*p*l+o*n*g-t*a*g)*M,e[15]=(o*f*s-u*a*s+u*n*l-t*f*l-o*n*d+t*a*d)*M,this}scale(e){const t=this.elements,n=e.x,s=e.y,r=e.z;return t[0]*=n,t[4]*=s,t[8]*=r,t[1]*=n,t[5]*=s,t[9]*=r,t[2]*=n,t[6]*=s,t[10]*=r,t[3]*=n,t[7]*=s,t[11]*=r,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,s))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const n=Math.cos(t),s=Math.sin(t),r=1-n,o=e.x,a=e.y,l=e.z,c=r*o,u=r*a;return this.set(c*o+n,c*a-s*l,c*l+s*a,0,c*a+s*l,u*a+n,u*l-s*o,0,c*l-s*a,u*l+s*o,r*l*l+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,s,r,o){return this.set(1,n,r,0,e,1,o,0,t,s,1,0,0,0,0,1),this}compose(e,t,n){const s=this.elements,r=t._x,o=t._y,a=t._z,l=t._w,c=r+r,u=o+o,f=a+a,d=r*c,h=r*u,x=r*f,p=o*u,g=o*f,m=a*f,_=l*c,v=l*u,A=l*f,S=n.x,y=n.y,M=n.z;return s[0]=(1-(p+m))*S,s[1]=(h+A)*S,s[2]=(x-v)*S,s[3]=0,s[4]=(h-A)*y,s[5]=(1-(d+m))*y,s[6]=(g+_)*y,s[7]=0,s[8]=(x+v)*M,s[9]=(g-_)*M,s[10]=(1-(d+p))*M,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,t,n){const s=this.elements;let r=ao.set(s[0],s[1],s[2]).length();const o=ao.set(s[4],s[5],s[6]).length(),a=ao.set(s[8],s[9],s[10]).length();this.determinant()<0&&(r=-r),e.x=s[12],e.y=s[13],e.z=s[14],Oi.copy(this);const c=1/r,u=1/o,f=1/a;return Oi.elements[0]*=c,Oi.elements[1]*=c,Oi.elements[2]*=c,Oi.elements[4]*=u,Oi.elements[5]*=u,Oi.elements[6]*=u,Oi.elements[8]*=f,Oi.elements[9]*=f,Oi.elements[10]*=f,t.setFromRotationMatrix(Oi),n.x=r,n.y=o,n.z=a,this}makePerspective(e,t,n,s,r,o,a=Ji,l=!1){const c=this.elements,u=2*r/(t-e),f=2*r/(n-s),d=(t+e)/(t-e),h=(n+s)/(n-s);let x,p;if(l)x=r/(o-r),p=o*r/(o-r);else if(a===Ji)x=-(o+r)/(o-r),p=-2*o*r/(o-r);else if(a===Ac)x=-o/(o-r),p=-o*r/(o-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=d,c[12]=0,c[1]=0,c[5]=f,c[9]=h,c[13]=0,c[2]=0,c[6]=0,c[10]=x,c[14]=p,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,t,n,s,r,o,a=Ji,l=!1){const c=this.elements,u=2/(t-e),f=2/(n-s),d=-(t+e)/(t-e),h=-(n+s)/(n-s);let x,p;if(l)x=1/(o-r),p=o/(o-r);else if(a===Ji)x=-2/(o-r),p=-(o+r)/(o-r);else if(a===Ac)x=-1/(o-r),p=-r/(o-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=0,c[12]=d,c[1]=0,c[5]=f,c[9]=0,c[13]=h,c[2]=0,c[6]=0,c[10]=x,c[14]=p,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){const t=this.elements,n=e.elements;for(let s=0;s<16;s++)if(t[s]!==n[s])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}}const ao=new U,Oi=new rt,ey=new U(0,0,0),ty=new U(1,1,1),Gs=new U,El=new U,li=new U,Up=new rt,Op=new Gt;class Gi{constructor(e=0,t=0,n=0,s=Gi.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=n,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,s=this._order){return this._x=e,this._y=t,this._z=n,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){const s=e.elements,r=s[0],o=s[4],a=s[8],l=s[1],c=s[5],u=s[9],f=s[2],d=s[6],h=s[10];switch(t){case"XYZ":this._y=Math.asin(ht(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-u,h),this._z=Math.atan2(-o,r)):(this._x=Math.atan2(d,c),this._z=0);break;case"YXZ":this._x=Math.asin(-ht(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(a,h),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-f,r),this._z=0);break;case"ZXY":this._x=Math.asin(ht(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-f,h),this._z=Math.atan2(-o,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-ht(f,-1,1)),Math.abs(f)<.9999999?(this._x=Math.atan2(d,h),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-o,c));break;case"YZX":this._z=Math.asin(ht(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-u,c),this._y=Math.atan2(-f,r)):(this._x=0,this._y=Math.atan2(a,h));break;case"XZY":this._z=Math.asin(-ht(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(d,c),this._y=Math.atan2(a,r)):(this._x=Math.atan2(-u,h),this._y=0);break;default:ut("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return Up.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Up,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return Op.setFromEuler(this),this.setFromQuaternion(Op,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}Gi.DEFAULT_ORDER="XYZ";class Hg{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let ny=0;const Np=new U,lo=new Gt,ms=new rt,wl=new U,pa=new U,iy=new U,sy=new Gt,zp=new U(1,0,0),kp=new U(0,1,0),Hp=new U(0,0,1),Vp={type:"added"},ry={type:"removed"},co={type:"childadded",child:null},Tu={type:"childremoved",child:null};class dn extends Kr{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:ny++}),this.uuid=ta(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=dn.DEFAULT_UP.clone();const e=new U,t=new Gi,n=new Gt,s=new U(1,1,1);function r(){n.setFromEuler(t,!1)}function o(){t.setFromQuaternion(n,void 0,!1)}t._onChange(r),n._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new rt},normalMatrix:{value:new at}}),this.matrix=new rt,this.matrixWorld=new rt,this.matrixAutoUpdate=dn.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=dn.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new Hg,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return lo.setFromAxisAngle(e,t),this.quaternion.multiply(lo),this}rotateOnWorldAxis(e,t){return lo.setFromAxisAngle(e,t),this.quaternion.premultiply(lo),this}rotateX(e){return this.rotateOnAxis(zp,e)}rotateY(e){return this.rotateOnAxis(kp,e)}rotateZ(e){return this.rotateOnAxis(Hp,e)}translateOnAxis(e,t){return Np.copy(e).applyQuaternion(this.quaternion),this.position.add(Np.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(zp,e)}translateY(e){return this.translateOnAxis(kp,e)}translateZ(e){return this.translateOnAxis(Hp,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(ms.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?wl.copy(e):wl.set(e,t,n);const s=this.parent;this.updateWorldMatrix(!0,!1),pa.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?ms.lookAt(pa,wl,this.up):ms.lookAt(wl,pa,this.up),this.quaternion.setFromRotationMatrix(ms),s&&(ms.extractRotation(s.matrixWorld),lo.setFromRotationMatrix(ms),this.quaternion.premultiply(lo.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(ln("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Vp),co.child=e,this.dispatchEvent(co),co.child=null):ln("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(ry),Tu.child=e,this.dispatchEvent(Tu),Tu.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),ms.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),ms.multiply(e.parent.matrixWorld)),e.applyMatrix4(ms),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Vp),co.child=e,this.dispatchEvent(co),co.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,s=this.children.length;n<s;n++){const o=this.children[n].getObjectByProperty(e,t);if(o!==void 0)return o}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(pa,e,iy),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(pa,sy,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let n=0,s=t.length;n<s;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t){const n=this.parent;if(e===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(a=>({...a,boundingBox:a.boundingBox?a.boundingBox.toJSON():void 0,boundingSphere:a.boundingSphere?a.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(a=>({...a})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(a,l){return a[l.uuid]===void 0&&(a[l.uuid]=l.toJSON(e)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);const a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){const l=a.shapes;if(Array.isArray(l))for(let c=0,u=l.length;c<u;c++){const f=l[c];r(e.shapes,f)}else r(e.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const a=[];for(let l=0,c=this.material.length;l<c;l++)a.push(r(e.materials,this.material[l]));s.material=a}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let a=0;a<this.children.length;a++)s.children.push(this.children[a].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let a=0;a<this.animations.length;a++){const l=this.animations[a];s.animations.push(r(e.animations,l))}}if(t){const a=o(e.geometries),l=o(e.materials),c=o(e.textures),u=o(e.images),f=o(e.shapes),d=o(e.skeletons),h=o(e.animations),x=o(e.nodes);a.length>0&&(n.geometries=a),l.length>0&&(n.materials=l),c.length>0&&(n.textures=c),u.length>0&&(n.images=u),f.length>0&&(n.shapes=f),d.length>0&&(n.skeletons=d),h.length>0&&(n.animations=h),x.length>0&&(n.nodes=x)}return n.object=s,n;function o(a){const l=[];for(const c in a){const u=a[c];delete u.metadata,l.push(u)}return l}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let n=0;n<e.children.length;n++){const s=e.children[n];this.add(s.clone())}return this}}dn.DEFAULT_UP=new U(0,1,0);dn.DEFAULT_MATRIX_AUTO_UPDATE=!0;dn.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const Ni=new U,gs=new U,Eu=new U,xs=new U,uo=new U,fo=new U,Gp=new U,wu=new U,Ru=new U,Iu=new U,Du=new Zt,Pu=new Zt,Fu=new Zt;class zi{constructor(e=new U,t=new U,n=new U){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,s){s.subVectors(n,t),Ni.subVectors(e,t),s.cross(Ni);const r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,t,n,s,r){Ni.subVectors(s,t),gs.subVectors(n,t),Eu.subVectors(e,t);const o=Ni.dot(Ni),a=Ni.dot(gs),l=Ni.dot(Eu),c=gs.dot(gs),u=gs.dot(Eu),f=o*c-a*a;if(f===0)return r.set(0,0,0),null;const d=1/f,h=(c*l-a*u)*d,x=(o*u-a*l)*d;return r.set(1-h-x,x,h)}static containsPoint(e,t,n,s){return this.getBarycoord(e,t,n,s,xs)===null?!1:xs.x>=0&&xs.y>=0&&xs.x+xs.y<=1}static getInterpolation(e,t,n,s,r,o,a,l){return this.getBarycoord(e,t,n,s,xs)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,xs.x),l.addScaledVector(o,xs.y),l.addScaledVector(a,xs.z),l)}static getInterpolatedAttribute(e,t,n,s,r,o){return Du.setScalar(0),Pu.setScalar(0),Fu.setScalar(0),Du.fromBufferAttribute(e,t),Pu.fromBufferAttribute(e,n),Fu.fromBufferAttribute(e,s),o.setScalar(0),o.addScaledVector(Du,r.x),o.addScaledVector(Pu,r.y),o.addScaledVector(Fu,r.z),o}static isFrontFacing(e,t,n,s){return Ni.subVectors(n,t),gs.subVectors(e,t),Ni.cross(gs).dot(s)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,s){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,t,n,s){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return Ni.subVectors(this.c,this.b),gs.subVectors(this.a,this.b),Ni.cross(gs).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return zi.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return zi.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,n,s,r){return zi.getInterpolation(e,this.a,this.b,this.c,t,n,s,r)}containsPoint(e){return zi.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return zi.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const n=this.a,s=this.b,r=this.c;let o,a;uo.subVectors(s,n),fo.subVectors(r,n),wu.subVectors(e,n);const l=uo.dot(wu),c=fo.dot(wu);if(l<=0&&c<=0)return t.copy(n);Ru.subVectors(e,s);const u=uo.dot(Ru),f=fo.dot(Ru);if(u>=0&&f<=u)return t.copy(s);const d=l*f-u*c;if(d<=0&&l>=0&&u<=0)return o=l/(l-u),t.copy(n).addScaledVector(uo,o);Iu.subVectors(e,r);const h=uo.dot(Iu),x=fo.dot(Iu);if(x>=0&&h<=x)return t.copy(r);const p=h*c-l*x;if(p<=0&&c>=0&&x<=0)return a=c/(c-x),t.copy(n).addScaledVector(fo,a);const g=u*x-h*f;if(g<=0&&f-u>=0&&h-x>=0)return Gp.subVectors(r,s),a=(f-u)/(f-u+(h-x)),t.copy(s).addScaledVector(Gp,a);const m=1/(g+p+d);return o=p*m,a=d*m,t.copy(n).addScaledVector(uo,o).addScaledVector(fo,a)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}const Vg={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Ws={h:0,s:0,l:0},Rl={h:0,s:0,l:0};function Lu(i,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?i+(e-i)*6*t:t<1/2?e:t<2/3?i+(e-i)*6*(2/3-t):i}class vt{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){const s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=Ci){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,yt.colorSpaceToWorking(this,t),this}setRGB(e,t,n,s=yt.workingColorSpace){return this.r=e,this.g=t,this.b=n,yt.colorSpaceToWorking(this,s),this}setHSL(e,t,n,s=yt.workingColorSpace){if(e=Jd(e,1),t=ht(t,0,1),n=ht(n,0,1),t===0)this.r=this.g=this.b=n;else{const r=n<=.5?n*(1+t):n+t-n*t,o=2*n-r;this.r=Lu(o,r,e+1/3),this.g=Lu(o,r,e),this.b=Lu(o,r,e-1/3)}return yt.colorSpaceToWorking(this,s),this}setStyle(e,t=Ci){function n(r){r!==void 0&&parseFloat(r)<1&&ut("Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r;const o=s[1],a=s[2];switch(o){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,t);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,t);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,t);break;default:ut("Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){const r=s[1],o=r.length;if(o===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,t);if(o===6)return this.setHex(parseInt(r,16),t);ut("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=Ci){const n=Vg[e.toLowerCase()];return n!==void 0?this.setHex(n,t):ut("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=Ps(e.r),this.g=Ps(e.g),this.b=Ps(e.b),this}copyLinearToSRGB(e){return this.r=Po(e.r),this.g=Po(e.g),this.b=Po(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Ci){return yt.workingToColorSpace(Dn.copy(this),e),Math.round(ht(Dn.r*255,0,255))*65536+Math.round(ht(Dn.g*255,0,255))*256+Math.round(ht(Dn.b*255,0,255))}getHexString(e=Ci){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=yt.workingColorSpace){yt.workingToColorSpace(Dn.copy(this),t);const n=Dn.r,s=Dn.g,r=Dn.b,o=Math.max(n,s,r),a=Math.min(n,s,r);let l,c;const u=(a+o)/2;if(a===o)l=0,c=0;else{const f=o-a;switch(c=u<=.5?f/(o+a):f/(2-o-a),o){case n:l=(s-r)/f+(s<r?6:0);break;case s:l=(r-n)/f+2;break;case r:l=(n-s)/f+4;break}l/=6}return e.h=l,e.s=c,e.l=u,e}getRGB(e,t=yt.workingColorSpace){return yt.workingToColorSpace(Dn.copy(this),t),e.r=Dn.r,e.g=Dn.g,e.b=Dn.b,e}getStyle(e=Ci){yt.workingToColorSpace(Dn.copy(this),e);const t=Dn.r,n=Dn.g,s=Dn.b;return e!==Ci?`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(s*255)})`}offsetHSL(e,t,n){return this.getHSL(Ws),this.setHSL(Ws.h+e,Ws.s+t,Ws.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(Ws),e.getHSL(Rl);const n=Da(Ws.h,Rl.h,t),s=Da(Ws.s,Rl.s,t),r=Da(Ws.l,Rl.l,t);return this.setHSL(n,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,n=this.g,s=this.b,r=e.elements;return this.r=r[0]*t+r[3]*n+r[6]*s,this.g=r[1]*t+r[4]*n+r[7]*s,this.b=r[2]*t+r[5]*n+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const Dn=new vt;vt.NAMES=Vg;let oy=0;class fl extends Kr{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:oy++}),this.uuid=ta(),this.name="",this.type="Material",this.blending=ir,this.side=as,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=Xa,this.blendDst=qa,this.blendEquation=Br,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new vt(0,0,0),this.blendAlpha=0,this.depthFunc=Ho,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Ep,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=no,this.stencilZFail=no,this.stencilZPass=no,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const n=e[t];if(n===void 0){ut(`Material: parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){ut(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(n):s&&s.isVector3&&n&&n.isVector3?s.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==ir&&(n.blending=this.blending),this.side!==as&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==Xa&&(n.blendSrc=this.blendSrc),this.blendDst!==qa&&(n.blendDst=this.blendDst),this.blendEquation!==Br&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==Ho&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Ep&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==no&&(n.stencilFail=this.stencilFail),this.stencilZFail!==no&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==no&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function s(r){const o=[];for(const a in r){const l=r[a];delete l.metadata,o.push(l)}return o}if(t){const r=s(e.textures),o=s(e.images);r.length>0&&(n.textures=r),o.length>0&&(n.images=o)}return n}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let n=null;if(t!==null){const s=t.length;n=new Array(s);for(let r=0;r!==s;++r)n[r]=t[r].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}class Yr extends fl{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new vt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Gi,this.combine=Rg,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const Rs=ay();function ay(){const i=new ArrayBuffer(4),e=new Float32Array(i),t=new Uint32Array(i),n=new Uint32Array(512),s=new Uint32Array(512);for(let l=0;l<256;++l){const c=l-127;c<-27?(n[l]=0,n[l|256]=32768,s[l]=24,s[l|256]=24):c<-14?(n[l]=1024>>-c-14,n[l|256]=1024>>-c-14|32768,s[l]=-c-1,s[l|256]=-c-1):c<=15?(n[l]=c+15<<10,n[l|256]=c+15<<10|32768,s[l]=13,s[l|256]=13):c<128?(n[l]=31744,n[l|256]=64512,s[l]=24,s[l|256]=24):(n[l]=31744,n[l|256]=64512,s[l]=13,s[l|256]=13)}const r=new Uint32Array(2048),o=new Uint32Array(64),a=new Uint32Array(64);for(let l=1;l<1024;++l){let c=l<<13,u=0;for(;(c&8388608)===0;)c<<=1,u-=8388608;c&=-8388609,u+=947912704,r[l]=c|u}for(let l=1024;l<2048;++l)r[l]=939524096+(l-1024<<13);for(let l=1;l<31;++l)o[l]=l<<23;o[31]=1199570944,o[32]=2147483648;for(let l=33;l<63;++l)o[l]=2147483648+(l-32<<23);o[63]=3347054592;for(let l=1;l<64;++l)l!==32&&(a[l]=1024);return{floatView:e,uint32View:t,baseTable:n,shiftTable:s,mantissaTable:r,exponentTable:o,offsetTable:a}}function ly(i){Math.abs(i)>65504&&ut("DataUtils.toHalfFloat(): Value out of range."),i=ht(i,-65504,65504),Rs.floatView[0]=i;const e=Rs.uint32View[0],t=e>>23&511;return Rs.baseTable[t]+((e&8388607)>>Rs.shiftTable[t])}function cy(i){const e=i>>10;return Rs.uint32View[0]=Rs.mantissaTable[Rs.offsetTable[e]+(i&1023)]+Rs.exponentTable[e],Rs.floatView[0]}class Za{static toHalfFloat(e){return ly(e)}static fromHalfFloat(e){return cy(e)}}const cn=new U,Il=new Ke;let uy=0;class Li{constructor(e,t,n=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:uy++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=n,this.usage=wp,this.updateRanges=[],this.gpuType=ki,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=t.array[n+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)Il.fromBufferAttribute(this,t),Il.applyMatrix3(e),this.setXY(t,Il.x,Il.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)cn.fromBufferAttribute(this,t),cn.applyMatrix3(e),this.setXYZ(t,cn.x,cn.y,cn.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)cn.fromBufferAttribute(this,t),cn.applyMatrix4(e),this.setXYZ(t,cn.x,cn.y,cn.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)cn.fromBufferAttribute(this,t),cn.applyNormalMatrix(e),this.setXYZ(t,cn.x,cn.y,cn.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)cn.fromBufferAttribute(this,t),cn.transformDirection(e),this.setXYZ(t,cn.x,cn.y,cn.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=So(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=Hn(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=So(t,this.array)),t}setX(e,t){return this.normalized&&(t=Hn(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=So(t,this.array)),t}setY(e,t){return this.normalized&&(t=Hn(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=So(t,this.array)),t}setZ(e,t){return this.normalized&&(t=Hn(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=So(t,this.array)),t}setW(e,t){return this.normalized&&(t=Hn(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=Hn(t,this.array),n=Hn(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,s){return e*=this.itemSize,this.normalized&&(t=Hn(t,this.array),n=Hn(n,this.array),s=Hn(s,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this}setXYZW(e,t,n,s,r){return e*=this.itemSize,this.normalized&&(t=Hn(t,this.array),n=Hn(n,this.array),s=Hn(s,this.array),r=Hn(r,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==wp&&(e.usage=this.usage),e}}class Gg extends Li{constructor(e,t,n){super(new Uint16Array(e),t,n)}}class Wg extends Li{constructor(e,t,n){super(new Uint32Array(e),t,n)}}class Bn extends Li{constructor(e,t,n){super(new Float32Array(e),t,n)}}let fy=0;const bi=new rt,Bu=new dn,ho=new U,ci=new es,ma=new es,_n=new U;class jn extends Kr{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:fy++}),this.uuid=ta(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(zg(e)?Wg:Gg)(e,1):this.index=e,this}setIndirect(e){return this.indirect=e,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const r=new at().getNormalMatrix(e);n.applyNormalMatrix(r),n.needsUpdate=!0}const s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return bi.makeRotationFromQuaternion(e),this.applyMatrix4(bi),this}rotateX(e){return bi.makeRotationX(e),this.applyMatrix4(bi),this}rotateY(e){return bi.makeRotationY(e),this.applyMatrix4(bi),this}rotateZ(e){return bi.makeRotationZ(e),this.applyMatrix4(bi),this}translate(e,t,n){return bi.makeTranslation(e,t,n),this.applyMatrix4(bi),this}scale(e,t,n){return bi.makeScale(e,t,n),this.applyMatrix4(bi),this}lookAt(e){return Bu.lookAt(e),Bu.updateMatrix(),this.applyMatrix4(Bu.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(ho).negate(),this.translate(ho.x,ho.y,ho.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const n=[];for(let s=0,r=e.length;s<r;s++){const o=e[s];n.push(o.x,o.y,o.z||0)}this.setAttribute("position",new Bn(n,3))}else{const n=Math.min(e.length,t.count);for(let s=0;s<n;s++){const r=e[s];t.setXYZ(s,r.x,r.y,r.z||0)}e.length>t.count&&ut("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new es);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){ln("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new U(-1/0,-1/0,-1/0),new U(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let n=0,s=t.length;n<s;n++){const r=t[n];ci.setFromBufferAttribute(r),this.morphTargetsRelative?(_n.addVectors(this.boundingBox.min,ci.min),this.boundingBox.expandByPoint(_n),_n.addVectors(this.boundingBox.max,ci.max),this.boundingBox.expandByPoint(_n)):(this.boundingBox.expandByPoint(ci.min),this.boundingBox.expandByPoint(ci.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&ln('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Xc);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){ln("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new U,1/0);return}if(e){const n=this.boundingSphere.center;if(ci.setFromBufferAttribute(e),t)for(let r=0,o=t.length;r<o;r++){const a=t[r];ma.setFromBufferAttribute(a),this.morphTargetsRelative?(_n.addVectors(ci.min,ma.min),ci.expandByPoint(_n),_n.addVectors(ci.max,ma.max),ci.expandByPoint(_n)):(ci.expandByPoint(ma.min),ci.expandByPoint(ma.max))}ci.getCenter(n);let s=0;for(let r=0,o=e.count;r<o;r++)_n.fromBufferAttribute(e,r),s=Math.max(s,n.distanceToSquared(_n));if(t)for(let r=0,o=t.length;r<o;r++){const a=t[r],l=this.morphTargetsRelative;for(let c=0,u=a.count;c<u;c++)_n.fromBufferAttribute(a,c),l&&(ho.fromBufferAttribute(e,c),_n.add(ho)),s=Math.max(s,n.distanceToSquared(_n))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&ln('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){ln("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=t.position,s=t.normal,r=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new Li(new Float32Array(4*n.count),4));const o=this.getAttribute("tangent"),a=[],l=[];for(let E=0;E<n.count;E++)a[E]=new U,l[E]=new U;const c=new U,u=new U,f=new U,d=new Ke,h=new Ke,x=new Ke,p=new U,g=new U;function m(E,b,C){c.fromBufferAttribute(n,E),u.fromBufferAttribute(n,b),f.fromBufferAttribute(n,C),d.fromBufferAttribute(r,E),h.fromBufferAttribute(r,b),x.fromBufferAttribute(r,C),u.sub(c),f.sub(c),h.sub(d),x.sub(d);const P=1/(h.x*x.y-x.x*h.y);isFinite(P)&&(p.copy(u).multiplyScalar(x.y).addScaledVector(f,-h.y).multiplyScalar(P),g.copy(f).multiplyScalar(h.x).addScaledVector(u,-x.x).multiplyScalar(P),a[E].add(p),a[b].add(p),a[C].add(p),l[E].add(g),l[b].add(g),l[C].add(g))}let _=this.groups;_.length===0&&(_=[{start:0,count:e.count}]);for(let E=0,b=_.length;E<b;++E){const C=_[E],P=C.start,B=C.count;for(let N=P,k=P+B;N<k;N+=3)m(e.getX(N+0),e.getX(N+1),e.getX(N+2))}const v=new U,A=new U,S=new U,y=new U;function M(E){S.fromBufferAttribute(s,E),y.copy(S);const b=a[E];v.copy(b),v.sub(S.multiplyScalar(S.dot(b))).normalize(),A.crossVectors(y,b);const P=A.dot(l[E])<0?-1:1;o.setXYZW(E,v.x,v.y,v.z,P)}for(let E=0,b=_.length;E<b;++E){const C=_[E],P=C.start,B=C.count;for(let N=P,k=P+B;N<k;N+=3)M(e.getX(N+0)),M(e.getX(N+1)),M(e.getX(N+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new Li(new Float32Array(t.count*3),3),this.setAttribute("normal",n);else for(let d=0,h=n.count;d<h;d++)n.setXYZ(d,0,0,0);const s=new U,r=new U,o=new U,a=new U,l=new U,c=new U,u=new U,f=new U;if(e)for(let d=0,h=e.count;d<h;d+=3){const x=e.getX(d+0),p=e.getX(d+1),g=e.getX(d+2);s.fromBufferAttribute(t,x),r.fromBufferAttribute(t,p),o.fromBufferAttribute(t,g),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),a.fromBufferAttribute(n,x),l.fromBufferAttribute(n,p),c.fromBufferAttribute(n,g),a.add(u),l.add(u),c.add(u),n.setXYZ(x,a.x,a.y,a.z),n.setXYZ(p,l.x,l.y,l.z),n.setXYZ(g,c.x,c.y,c.z)}else for(let d=0,h=t.count;d<h;d+=3)s.fromBufferAttribute(t,d+0),r.fromBufferAttribute(t,d+1),o.fromBufferAttribute(t,d+2),u.subVectors(o,r),f.subVectors(s,r),u.cross(f),n.setXYZ(d+0,u.x,u.y,u.z),n.setXYZ(d+1,u.x,u.y,u.z),n.setXYZ(d+2,u.x,u.y,u.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)_n.fromBufferAttribute(e,t),_n.normalize(),e.setXYZ(t,_n.x,_n.y,_n.z)}toNonIndexed(){function e(a,l){const c=a.array,u=a.itemSize,f=a.normalized,d=new c.constructor(l.length*u);let h=0,x=0;for(let p=0,g=l.length;p<g;p++){a.isInterleavedBufferAttribute?h=l[p]*a.data.stride+a.offset:h=l[p]*u;for(let m=0;m<u;m++)d[x++]=c[h++]}return new Li(d,u,f)}if(this.index===null)return ut("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new jn,n=this.index.array,s=this.attributes;for(const a in s){const l=s[a],c=e(l,n);t.setAttribute(a,c)}const r=this.morphAttributes;for(const a in r){const l=[],c=r[a];for(let u=0,f=c.length;u<f;u++){const d=c[u],h=e(d,n);l.push(h)}t.morphAttributes[a]=l}t.morphTargetsRelative=this.morphTargetsRelative;const o=this.groups;for(let a=0,l=o.length;a<l;a++){const c=o[a];t.addGroup(c.start,c.count,c.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const l=this.parameters;for(const c in l)l[c]!==void 0&&(e[c]=l[c]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const n=this.attributes;for(const l in n){const c=n[l];e.data.attributes[l]=c.toJSON(e.data)}const s={};let r=!1;for(const l in this.morphAttributes){const c=this.morphAttributes[l],u=[];for(let f=0,d=c.length;f<d;f++){const h=c[f];u.push(h.toJSON(e.data))}u.length>0&&(s[l]=u,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);const o=this.groups;o.length>0&&(e.data.groups=JSON.parse(JSON.stringify(o)));const a=this.boundingSphere;return a!==null&&(e.data.boundingSphere=a.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const n=e.index;n!==null&&this.setIndex(n.clone());const s=e.attributes;for(const c in s){const u=s[c];this.setAttribute(c,u.clone(t))}const r=e.morphAttributes;for(const c in r){const u=[],f=r[c];for(let d=0,h=f.length;d<h;d++)u.push(f[d].clone(t));this.morphAttributes[c]=u}this.morphTargetsRelative=e.morphTargetsRelative;const o=e.groups;for(let c=0,u=o.length;c<u;c++){const f=o[c];this.addGroup(f.start,f.count,f.materialIndex)}const a=e.boundingBox;a!==null&&(this.boundingBox=a.clone());const l=e.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const Wp=new rt,yr=new th,Dl=new Xc,Xp=new U,Pl=new U,Fl=new U,Ll=new U,Uu=new U,Bl=new U,qp=new U,Ul=new U;class un extends dn{constructor(e=new jn,t=new Yr){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}getVertexPosition(e,t){const n=this.geometry,s=n.attributes.position,r=n.morphAttributes.position,o=n.morphTargetsRelative;t.fromBufferAttribute(s,e);const a=this.morphTargetInfluences;if(r&&a){Bl.set(0,0,0);for(let l=0,c=r.length;l<c;l++){const u=a[l],f=r[l];u!==0&&(Uu.fromBufferAttribute(f,e),o?Bl.addScaledVector(Uu,u):Bl.addScaledVector(Uu.sub(t),u))}t.add(Bl)}return t}raycast(e,t){const n=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),Dl.copy(n.boundingSphere),Dl.applyMatrix4(r),yr.copy(e.ray).recast(e.near),!(Dl.containsPoint(yr.origin)===!1&&(yr.intersectSphere(Dl,Xp)===null||yr.origin.distanceToSquared(Xp)>(e.far-e.near)**2))&&(Wp.copy(r).invert(),yr.copy(e.ray).applyMatrix4(Wp),!(n.boundingBox!==null&&yr.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,yr)))}_computeIntersections(e,t,n){let s;const r=this.geometry,o=this.material,a=r.index,l=r.attributes.position,c=r.attributes.uv,u=r.attributes.uv1,f=r.attributes.normal,d=r.groups,h=r.drawRange;if(a!==null)if(Array.isArray(o))for(let x=0,p=d.length;x<p;x++){const g=d[x],m=o[g.materialIndex],_=Math.max(g.start,h.start),v=Math.min(a.count,Math.min(g.start+g.count,h.start+h.count));for(let A=_,S=v;A<S;A+=3){const y=a.getX(A),M=a.getX(A+1),E=a.getX(A+2);s=Ol(this,m,e,n,c,u,f,y,M,E),s&&(s.faceIndex=Math.floor(A/3),s.face.materialIndex=g.materialIndex,t.push(s))}}else{const x=Math.max(0,h.start),p=Math.min(a.count,h.start+h.count);for(let g=x,m=p;g<m;g+=3){const _=a.getX(g),v=a.getX(g+1),A=a.getX(g+2);s=Ol(this,o,e,n,c,u,f,_,v,A),s&&(s.faceIndex=Math.floor(g/3),t.push(s))}}else if(l!==void 0)if(Array.isArray(o))for(let x=0,p=d.length;x<p;x++){const g=d[x],m=o[g.materialIndex],_=Math.max(g.start,h.start),v=Math.min(l.count,Math.min(g.start+g.count,h.start+h.count));for(let A=_,S=v;A<S;A+=3){const y=A,M=A+1,E=A+2;s=Ol(this,m,e,n,c,u,f,y,M,E),s&&(s.faceIndex=Math.floor(A/3),s.face.materialIndex=g.materialIndex,t.push(s))}}else{const x=Math.max(0,h.start),p=Math.min(l.count,h.start+h.count);for(let g=x,m=p;g<m;g+=3){const _=g,v=g+1,A=g+2;s=Ol(this,o,e,n,c,u,f,_,v,A),s&&(s.faceIndex=Math.floor(g/3),t.push(s))}}}}function dy(i,e,t,n,s,r,o,a){let l;if(e.side===ti?l=n.intersectTriangle(o,r,s,!0,a):l=n.intersectTriangle(s,r,o,e.side===as,a),l===null)return null;Ul.copy(a),Ul.applyMatrix4(i.matrixWorld);const c=t.ray.origin.distanceTo(Ul);return c<t.near||c>t.far?null:{distance:c,point:Ul.clone(),object:i}}function Ol(i,e,t,n,s,r,o,a,l,c){i.getVertexPosition(a,Pl),i.getVertexPosition(l,Fl),i.getVertexPosition(c,Ll);const u=dy(i,e,t,n,Pl,Fl,Ll,qp);if(u){const f=new U;zi.getBarycoord(qp,Pl,Fl,Ll,f),s&&(u.uv=zi.getInterpolatedAttribute(s,a,l,c,f,new Ke)),r&&(u.uv1=zi.getInterpolatedAttribute(r,a,l,c,f,new Ke)),o&&(u.normal=zi.getInterpolatedAttribute(o,a,l,c,f,new U),u.normal.dot(n.direction)>0&&u.normal.multiplyScalar(-1));const d={a,b:l,c,normal:new U,materialIndex:0};zi.getNormal(Pl,Fl,Ll,d.normal),u.face=d,u.barycoord=f}return u}class na extends jn{constructor(e=1,t=1,n=1,s=1,r=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:n,widthSegments:s,heightSegments:r,depthSegments:o};const a=this;s=Math.floor(s),r=Math.floor(r),o=Math.floor(o);const l=[],c=[],u=[],f=[];let d=0,h=0;x("z","y","x",-1,-1,n,t,e,o,r,0),x("z","y","x",1,-1,n,t,-e,o,r,1),x("x","z","y",1,1,e,n,t,s,o,2),x("x","z","y",1,-1,e,n,-t,s,o,3),x("x","y","z",1,-1,e,t,n,s,r,4),x("x","y","z",-1,-1,e,t,-n,s,r,5),this.setIndex(l),this.setAttribute("position",new Bn(c,3)),this.setAttribute("normal",new Bn(u,3)),this.setAttribute("uv",new Bn(f,2));function x(p,g,m,_,v,A,S,y,M,E,b){const C=A/M,P=S/E,B=A/2,N=S/2,k=y/2,V=M+1,F=E+1;let q=0,G=0;const j=new U;for(let ce=0;ce<F;ce++){const ye=ce*P-N;for(let Ie=0;Ie<V;Ie++){const Ne=Ie*C-B;j[p]=Ne*_,j[g]=ye*v,j[m]=k,c.push(j.x,j.y,j.z),j[p]=0,j[g]=0,j[m]=y>0?1:-1,u.push(j.x,j.y,j.z),f.push(Ie/M),f.push(1-ce/E),q+=1}}for(let ce=0;ce<E;ce++)for(let ye=0;ye<M;ye++){const Ie=d+ye+V*ce,Ne=d+ye+V*(ce+1),de=d+(ye+1)+V*(ce+1),qe=d+(ye+1)+V*ce;l.push(Ie,Ne,qe),l.push(Ne,de,qe),G+=6}a.addGroup(h,G,b),h+=G,d+=q}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new na(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}function qo(i){const e={};for(const t in i){e[t]={};for(const n in i[t]){const s=i[t][n];s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)?s.isRenderTargetTexture?(ut("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][n]=null):e[t][n]=s.clone():Array.isArray(s)?e[t][n]=s.slice():e[t][n]=s}}return e}function Vn(i){const e={};for(let t=0;t<i.length;t++){const n=qo(i[t]);for(const s in n)e[s]=n[s]}return e}function hy(i){const e=[];for(let t=0;t<i.length;t++)e.push(i[t].clone());return e}function Xg(i){const e=i.getRenderTarget();return e===null?i.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:yt.workingColorSpace}const py={clone:qo,merge:Vn};var my=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,gy=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class Kn extends fl{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=my,this.fragmentShader=gy,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=qo(e.uniforms),this.uniformsGroups=hy(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const s in this.uniforms){const o=this.uniforms[s].value;o&&o.isTexture?t.uniforms[s]={type:"t",value:o.toJSON(e).uuid}:o&&o.isColor?t.uniforms[s]={type:"c",value:o.getHex()}:o&&o.isVector2?t.uniforms[s]={type:"v2",value:o.toArray()}:o&&o.isVector3?t.uniforms[s]={type:"v3",value:o.toArray()}:o&&o.isVector4?t.uniforms[s]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?t.uniforms[s]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?t.uniforms[s]={type:"m4",value:o.toArray()}:t.uniforms[s]={value:o}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const n={};for(const s in this.extensions)this.extensions[s]===!0&&(n[s]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}}class qg extends dn{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new rt,this.projectionMatrix=new rt,this.projectionMatrixInverse=new rt,this.coordinateSystem=Ji,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const Xs=new U,Yp=new Ke,Qp=new Ke;class Ti extends qg{constructor(e=50,t=1,n=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=n,this.far=s,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=$a*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(Ia*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return $a*2*Math.atan(Math.tan(Ia*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){Xs.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(Xs.x,Xs.y).multiplyScalar(-e/Xs.z),Xs.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(Xs.x,Xs.y).multiplyScalar(-e/Xs.z)}getViewSize(e,t){return this.getViewBounds(e,Yp,Qp),t.subVectors(Qp,Yp)}setViewOffset(e,t,n,s,r,o){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(Ia*.5*this.fov)/this.zoom,n=2*t,s=this.aspect*n,r=-.5*s;const o=this.view;if(this.view!==null&&this.view.enabled){const l=o.fullWidth,c=o.fullHeight;r+=o.offsetX*s/l,t-=o.offsetY*n/c,s*=o.width/l,n*=o.height/c}const a=this.filmOffset;a!==0&&(r+=e*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}const po=-90,mo=1;class xy extends dn{constructor(e,t,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const s=new Ti(po,mo,e,t);s.layers=this.layers,this.add(s);const r=new Ti(po,mo,e,t);r.layers=this.layers,this.add(r);const o=new Ti(po,mo,e,t);o.layers=this.layers,this.add(o);const a=new Ti(po,mo,e,t);a.layers=this.layers,this.add(a);const l=new Ti(po,mo,e,t);l.layers=this.layers,this.add(l);const c=new Ti(po,mo,e,t);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[n,s,r,o,a,l]=t;for(const c of t)this.remove(c);if(e===Ji)n.up.set(0,1,0),n.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),o.up.set(0,0,1),o.lookAt(0,-1,0),a.up.set(0,1,0),a.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(e===Ac)n.up.set(0,-1,0),n.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),o.up.set(0,0,-1),o.lookAt(0,-1,0),a.up.set(0,-1,0),a.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const c of t)this.add(c),c.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[r,o,a,l,c,u]=this.children,f=e.getRenderTarget(),d=e.getActiveCubeFace(),h=e.getActiveMipmapLevel(),x=e.xr.enabled;e.xr.enabled=!1;const p=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,e.setRenderTarget(n,0,s),e.render(t,r),e.setRenderTarget(n,1,s),e.render(t,o),e.setRenderTarget(n,2,s),e.render(t,a),e.setRenderTarget(n,3,s),e.render(t,l),e.setRenderTarget(n,4,s),e.render(t,c),n.texture.generateMipmaps=p,e.setRenderTarget(n,5,s),e.render(t,u),e.setRenderTarget(f,d,h),e.xr.enabled=x,n.texture.needsPMREMUpdate=!0}}class Yg extends Qn{constructor(e=[],t=Vo,n,s,r,o,a,l,c,u){super(e,t,n,s,r,o,a,l,c,u),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class _y extends cr{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const n={width:e,height:e,depth:1},s=[n,n,n,n,n,n];this.texture=new Yg(s),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},s=new na(5,5,5),r=new Kn({name:"CubemapFromEquirect",uniforms:qo(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:ti,blending:Ds});r.uniforms.tEquirect.value=t;const o=new un(s,r),a=t.minFilter;return t.minFilter===Nr&&(t.minFilter=Ri),new xy(1,10,this).update(e,o),t.minFilter=a,o.geometry.dispose(),o.material.dispose(),this}clear(e,t=!0,n=!0,s=!0){const r=e.getRenderTarget();for(let o=0;o<6;o++)e.setRenderTarget(this,o),e.clear(t,n,s);e.setRenderTarget(r)}}class Nl extends dn{constructor(){super(),this.isGroup=!0,this.type="Group"}}const vy={type:"move"};class Ou{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Nl,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Nl,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new U,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new U),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Nl,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new U,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new U),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let s=null,r=null,o=null;const a=this._targetRay,l=this._grip,c=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(c&&e.hand){o=!0;for(const p of e.hand.values()){const g=t.getJointPose(p,n),m=this._getHandJoint(c,p);g!==null&&(m.matrix.fromArray(g.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=g.radius),m.visible=g!==null}const u=c.joints["index-finger-tip"],f=c.joints["thumb-tip"],d=u.position.distanceTo(f.position),h=.02,x=.005;c.inputState.pinching&&d>h+x?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!c.inputState.pinching&&d<=h-x&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else l!==null&&e.gripSpace&&(r=t.getPose(e.gripSpace,n),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.rotation,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1));a!==null&&(s=t.getPose(e.targetRaySpace,n),s===null&&r!==null&&(s=r),s!==null&&(a.matrix.fromArray(s.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,s.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(s.linearVelocity)):a.hasLinearVelocity=!1,s.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(s.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(vy)))}return a!==null&&(a.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=o!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const n=new Nl;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}}class Ay extends dn{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Gi,this.environmentIntensity=1,this.environmentRotation=new Gi,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}class Ss extends Qn{constructor(e=null,t=1,n=1,s,r,o,a,l,c=xi,u=xi,f,d){super(null,o,a,l,c,u,s,r,f,d),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class Sy extends Li{constructor(e,t,n,s=1){super(e,t,n),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}toJSON(){const e=super.toJSON();return e.meshPerAttribute=this.meshPerAttribute,e.isInstancedBufferAttribute=!0,e}}const Nu=new U,yy=new U,by=new at;class Ks{constructor(e=new U(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,s){return this.normal.set(e,t,n),this.constant=s,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){const s=Nu.subVectors(n,t).cross(yy.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t){const n=e.delta(Nu),s=this.normal.dot(n);if(s===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const r=-(e.start.dot(this.normal)+this.constant)/s;return r<0||r>1?null:t.copy(e.start).addScaledVector(n,r)}intersectsLine(e){const t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const n=t||by.getNormalMatrix(e),s=this.coplanarPoint(Nu).applyMatrix4(e),r=this.normal.applyMatrix3(n).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const br=new Xc,My=new Ke(.5,.5),zl=new U;class Qg{constructor(e=new Ks,t=new Ks,n=new Ks,s=new Ks,r=new Ks,o=new Ks){this.planes=[e,t,n,s,r,o]}set(e,t,n,s,r,o){const a=this.planes;return a[0].copy(e),a[1].copy(t),a[2].copy(n),a[3].copy(s),a[4].copy(r),a[5].copy(o),this}copy(e){const t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=Ji,n=!1){const s=this.planes,r=e.elements,o=r[0],a=r[1],l=r[2],c=r[3],u=r[4],f=r[5],d=r[6],h=r[7],x=r[8],p=r[9],g=r[10],m=r[11],_=r[12],v=r[13],A=r[14],S=r[15];if(s[0].setComponents(c-o,h-u,m-x,S-_).normalize(),s[1].setComponents(c+o,h+u,m+x,S+_).normalize(),s[2].setComponents(c+a,h+f,m+p,S+v).normalize(),s[3].setComponents(c-a,h-f,m-p,S-v).normalize(),n)s[4].setComponents(l,d,g,A).normalize(),s[5].setComponents(c-l,h-d,m-g,S-A).normalize();else if(s[4].setComponents(c-l,h-d,m-g,S-A).normalize(),t===Ji)s[5].setComponents(c+l,h+d,m+g,S+A).normalize();else if(t===Ac)s[5].setComponents(l,d,g,A).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),br.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),br.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(br)}intersectsSprite(e){br.center.set(0,0,0);const t=My.distanceTo(e.center);return br.radius=.7071067811865476+t,br.applyMatrix4(e.matrixWorld),this.intersectsSphere(br)}intersectsSphere(e){const t=this.planes,n=e.center,s=-e.radius;for(let r=0;r<6;r++)if(t[r].distanceToPoint(n)<s)return!1;return!0}intersectsBox(e){const t=this.planes;for(let n=0;n<6;n++){const s=t[n];if(zl.x=s.normal.x>0?e.max.x:e.min.x,zl.y=s.normal.y>0?e.max.y:e.min.y,zl.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(zl)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class Cy extends fl{constructor(e){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new vt(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.size=e.size,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}}const Kp=new rt,ud=new th,kl=new Xc,Hl=new U;class Ty extends dn{constructor(e=new jn,t=new Cy){super(),this.isPoints=!0,this.type="Points",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}raycast(e,t){const n=this.geometry,s=this.matrixWorld,r=e.params.Points.threshold,o=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),kl.copy(n.boundingSphere),kl.applyMatrix4(s),kl.radius+=r,e.ray.intersectsSphere(kl)===!1)return;Kp.copy(s).invert(),ud.copy(e.ray).applyMatrix4(Kp);const a=r/((this.scale.x+this.scale.y+this.scale.z)/3),l=a*a,c=n.index,f=n.attributes.position;if(c!==null){const d=Math.max(0,o.start),h=Math.min(c.count,o.start+o.count);for(let x=d,p=h;x<p;x++){const g=c.getX(x);Hl.fromBufferAttribute(f,g),jp(Hl,g,l,s,e,t,this)}}else{const d=Math.max(0,o.start),h=Math.min(f.count,o.start+o.count);for(let x=d,p=h;x<p;x++)Hl.fromBufferAttribute(f,x),jp(Hl,x,l,s,e,t,this)}}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const s=t[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}}function jp(i,e,t,n,s,r,o){const a=ud.distanceSqToPoint(i);if(a<t){const l=new U;ud.closestPointToPoint(i,l),l.applyMatrix4(n);const c=s.ray.origin.distanceTo(l);if(c<s.near||c>s.far)return;r.push({distance:c,distanceToRay:Math.sqrt(a),point:l,index:e,face:null,faceIndex:null,barycoord:null,object:o})}}class nh extends Qn{constructor(e,t,n=Ii,s,r,o,a=xi,l=xi,c,u=Wo,f=1){if(u!==Wo&&u!==Ka)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const d={width:e,height:t,depth:f};super(d,s,r,o,a,l,u,n,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new eh(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class Kg extends Qn{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class Ja extends jn{constructor(e=1,t=1,n=1,s=32,r=1,o=!1,a=0,l=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:n,radialSegments:s,heightSegments:r,openEnded:o,thetaStart:a,thetaLength:l};const c=this;s=Math.floor(s),r=Math.floor(r);const u=[],f=[],d=[],h=[];let x=0;const p=[],g=n/2;let m=0;_(),o===!1&&(e>0&&v(!0),t>0&&v(!1)),this.setIndex(u),this.setAttribute("position",new Bn(f,3)),this.setAttribute("normal",new Bn(d,3)),this.setAttribute("uv",new Bn(h,2));function _(){const A=new U,S=new U;let y=0;const M=(t-e)/n;for(let E=0;E<=r;E++){const b=[],C=E/r,P=C*(t-e)+e;for(let B=0;B<=s;B++){const N=B/s,k=N*l+a,V=Math.sin(k),F=Math.cos(k);S.x=P*V,S.y=-C*n+g,S.z=P*F,f.push(S.x,S.y,S.z),A.set(V,M,F).normalize(),d.push(A.x,A.y,A.z),h.push(N,1-C),b.push(x++)}p.push(b)}for(let E=0;E<s;E++)for(let b=0;b<r;b++){const C=p[b][E],P=p[b+1][E],B=p[b+1][E+1],N=p[b][E+1];(e>0||b!==0)&&(u.push(C,P,N),y+=3),(t>0||b!==r-1)&&(u.push(P,B,N),y+=3)}c.addGroup(m,y,0),m+=y}function v(A){const S=x,y=new Ke,M=new U;let E=0;const b=A===!0?e:t,C=A===!0?1:-1;for(let B=1;B<=s;B++)f.push(0,g*C,0),d.push(0,C,0),h.push(.5,.5),x++;const P=x;for(let B=0;B<=s;B++){const k=B/s*l+a,V=Math.cos(k),F=Math.sin(k);M.x=b*F,M.y=g*C,M.z=b*V,f.push(M.x,M.y,M.z),d.push(0,C,0),y.x=V*.5+.5,y.y=F*.5*C+.5,h.push(y.x,y.y),x++}for(let B=0;B<s;B++){const N=S+B,k=P+B;A===!0?u.push(k,k+1,N):u.push(k+1,k,N),E+=3}c.addGroup(m,E,A===!0?1:2),m+=E}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Ja(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class ih extends Ja{constructor(e=1,t=1,n=32,s=1,r=!1,o=0,a=Math.PI*2){super(0,e,t,n,s,r,o,a),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:n,heightSegments:s,openEnded:r,thetaStart:o,thetaLength:a}}static fromJSON(e){return new ih(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class Ey{constructor(){this.type="Curve",this.arcLengthDivisions=200,this.needsUpdate=!1,this.cacheArcLengths=null}getPoint(){ut("Curve: .getPoint() not implemented.")}getPointAt(e,t){const n=this.getUtoTmapping(e);return this.getPoint(n,t)}getPoints(e=5){const t=[];for(let n=0;n<=e;n++)t.push(this.getPoint(n/e));return t}getSpacedPoints(e=5){const t=[];for(let n=0;n<=e;n++)t.push(this.getPointAt(n/e));return t}getLength(){const e=this.getLengths();return e[e.length-1]}getLengths(e=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===e+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;const t=[];let n,s=this.getPoint(0),r=0;t.push(0);for(let o=1;o<=e;o++)n=this.getPoint(o/e),r+=n.distanceTo(s),t.push(r),s=n;return this.cacheArcLengths=t,t}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(e,t=null){const n=this.getLengths();let s=0;const r=n.length;let o;t?o=t:o=e*n[r-1];let a=0,l=r-1,c;for(;a<=l;)if(s=Math.floor(a+(l-a)/2),c=n[s]-o,c<0)a=s+1;else if(c>0)l=s-1;else{l=s;break}if(s=l,n[s]===o)return s/(r-1);const u=n[s],d=n[s+1]-u,h=(o-u)/d;return(s+h)/(r-1)}getTangent(e,t){let s=e-1e-4,r=e+1e-4;s<0&&(s=0),r>1&&(r=1);const o=this.getPoint(s),a=this.getPoint(r),l=t||(o.isVector2?new Ke:new U);return l.copy(a).sub(o).normalize(),l}getTangentAt(e,t){const n=this.getUtoTmapping(e);return this.getTangent(n,t)}computeFrenetFrames(e,t=!1){const n=new U,s=[],r=[],o=[],a=new U,l=new rt;for(let h=0;h<=e;h++){const x=h/e;s[h]=this.getTangentAt(x,new U)}r[0]=new U,o[0]=new U;let c=Number.MAX_VALUE;const u=Math.abs(s[0].x),f=Math.abs(s[0].y),d=Math.abs(s[0].z);u<=c&&(c=u,n.set(1,0,0)),f<=c&&(c=f,n.set(0,1,0)),d<=c&&n.set(0,0,1),a.crossVectors(s[0],n).normalize(),r[0].crossVectors(s[0],a),o[0].crossVectors(s[0],r[0]);for(let h=1;h<=e;h++){if(r[h]=r[h-1].clone(),o[h]=o[h-1].clone(),a.crossVectors(s[h-1],s[h]),a.length()>Number.EPSILON){a.normalize();const x=Math.acos(ht(s[h-1].dot(s[h]),-1,1));r[h].applyMatrix4(l.makeRotationAxis(a,x))}o[h].crossVectors(s[h],r[h])}if(t===!0){let h=Math.acos(ht(r[0].dot(r[e]),-1,1));h/=e,s[0].dot(a.crossVectors(r[0],r[e]))>0&&(h=-h);for(let x=1;x<=e;x++)r[x].applyMatrix4(l.makeRotationAxis(s[x],h*x)),o[x].crossVectors(s[x],r[x])}return{tangents:s,normals:r,binormals:o}}clone(){return new this.constructor().copy(this)}copy(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}toJSON(){const e={metadata:{version:4.7,type:"Curve",generator:"Curve.toJSON"}};return e.arcLengthDivisions=this.arcLengthDivisions,e.type=this.type,e}fromJSON(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}}function sh(){let i=0,e=0,t=0,n=0;function s(r,o,a,l){i=r,e=a,t=-3*r+3*o-2*a-l,n=2*r-2*o+a+l}return{initCatmullRom:function(r,o,a,l,c){s(o,a,c*(a-r),c*(l-o))},initNonuniformCatmullRom:function(r,o,a,l,c,u,f){let d=(o-r)/c-(a-r)/(c+u)+(a-o)/u,h=(a-o)/u-(l-o)/(u+f)+(l-a)/f;d*=u,h*=u,s(o,a,d,h)},calc:function(r){const o=r*r,a=o*r;return i+e*r+t*o+n*a}}}const Vl=new U,zu=new sh,ku=new sh,Hu=new sh;class $p extends Ey{constructor(e=[],t=!1,n="centripetal",s=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=e,this.closed=t,this.curveType=n,this.tension=s}getPoint(e,t=new U){const n=t,s=this.points,r=s.length,o=(r-(this.closed?0:1))*e;let a=Math.floor(o),l=o-a;this.closed?a+=a>0?0:(Math.floor(Math.abs(a)/r)+1)*r:l===0&&a===r-1&&(a=r-2,l=1);let c,u;this.closed||a>0?c=s[(a-1)%r]:(Vl.subVectors(s[0],s[1]).add(s[0]),c=Vl);const f=s[a%r],d=s[(a+1)%r];if(this.closed||a+2<r?u=s[(a+2)%r]:(Vl.subVectors(s[r-1],s[r-2]).add(s[r-1]),u=Vl),this.curveType==="centripetal"||this.curveType==="chordal"){const h=this.curveType==="chordal"?.5:.25;let x=Math.pow(c.distanceToSquared(f),h),p=Math.pow(f.distanceToSquared(d),h),g=Math.pow(d.distanceToSquared(u),h);p<1e-4&&(p=1),x<1e-4&&(x=p),g<1e-4&&(g=p),zu.initNonuniformCatmullRom(c.x,f.x,d.x,u.x,x,p,g),ku.initNonuniformCatmullRom(c.y,f.y,d.y,u.y,x,p,g),Hu.initNonuniformCatmullRom(c.z,f.z,d.z,u.z,x,p,g)}else this.curveType==="catmullrom"&&(zu.initCatmullRom(c.x,f.x,d.x,u.x,this.tension),ku.initCatmullRom(c.y,f.y,d.y,u.y,this.tension),Hu.initCatmullRom(c.z,f.z,d.z,u.z,this.tension));return n.set(zu.calc(l),ku.calc(l),Hu.calc(l)),n}copy(e){super.copy(e),this.points=[];for(let t=0,n=e.points.length;t<n;t++){const s=e.points[t];this.points.push(s.clone())}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}toJSON(){const e=super.toJSON();e.points=[];for(let t=0,n=this.points.length;t<n;t++){const s=this.points[t];e.points.push(s.toArray())}return e.closed=this.closed,e.curveType=this.curveType,e.tension=this.tension,e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,n=e.points.length;t<n;t++){const s=e.points[t];this.points.push(new U().fromArray(s))}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}}class Yo extends jn{constructor(e=1,t=1,n=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:n,heightSegments:s};const r=e/2,o=t/2,a=Math.floor(n),l=Math.floor(s),c=a+1,u=l+1,f=e/a,d=t/l,h=[],x=[],p=[],g=[];for(let m=0;m<u;m++){const _=m*d-o;for(let v=0;v<c;v++){const A=v*f-r;x.push(A,-_,0),p.push(0,0,1),g.push(v/a),g.push(1-m/l)}}for(let m=0;m<l;m++)for(let _=0;_<a;_++){const v=_+c*m,A=_+c*(m+1),S=_+1+c*(m+1),y=_+1+c*m;h.push(v,A,y),h.push(A,S,y)}this.setIndex(h),this.setAttribute("position",new Bn(x,3)),this.setAttribute("normal",new Bn(p,3)),this.setAttribute("uv",new Bn(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Yo(e.width,e.height,e.widthSegments,e.heightSegments)}}class yc extends jn{constructor(e=1,t=32,n=16,s=0,r=Math.PI*2,o=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:e,widthSegments:t,heightSegments:n,phiStart:s,phiLength:r,thetaStart:o,thetaLength:a},t=Math.max(3,Math.floor(t)),n=Math.max(2,Math.floor(n));const l=Math.min(o+a,Math.PI);let c=0;const u=[],f=new U,d=new U,h=[],x=[],p=[],g=[];for(let m=0;m<=n;m++){const _=[],v=m/n;let A=0;m===0&&o===0?A=.5/t:m===n&&l===Math.PI&&(A=-.5/t);for(let S=0;S<=t;S++){const y=S/t;f.x=-e*Math.cos(s+y*r)*Math.sin(o+v*a),f.y=e*Math.cos(o+v*a),f.z=e*Math.sin(s+y*r)*Math.sin(o+v*a),x.push(f.x,f.y,f.z),d.copy(f).normalize(),p.push(d.x,d.y,d.z),g.push(y+A,1-v),_.push(c++)}u.push(_)}for(let m=0;m<n;m++)for(let _=0;_<t;_++){const v=u[m][_+1],A=u[m][_],S=u[m+1][_],y=u[m+1][_+1];(m!==0||o>0)&&h.push(v,A,y),(m!==n-1||l<Math.PI)&&h.push(A,S,y)}this.setIndex(h),this.setAttribute("position",new Bn(x,3)),this.setAttribute("normal",new Bn(p,3)),this.setAttribute("uv",new Bn(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new yc(e.radius,e.widthSegments,e.heightSegments,e.phiStart,e.phiLength,e.thetaStart,e.thetaLength)}}class wy extends fl{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=xS,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class Ry extends fl{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}class rh extends qg{constructor(e=-1,t=1,n=1,s=-1,r=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=s,this.near=r,this.far=o,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,s,r,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,s=(this.top+this.bottom)/2;let r=n-e,o=n+e,a=s+t,l=s-t;if(this.view!==null&&this.view.enabled){const c=(this.right-this.left)/this.view.fullWidth/this.zoom,u=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,o=r+c*this.view.width,a-=u*this.view.offsetY,l=a-u*this.view.height}this.projectionMatrix.makeOrthographic(r,o,a,l,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}class Iy extends jn{constructor(){super(),this.isInstancedBufferGeometry=!0,this.type="InstancedBufferGeometry",this.instanceCount=1/0}copy(e){return super.copy(e),this.instanceCount=e.instanceCount,this}toJSON(){const e=super.toJSON();return e.instanceCount=this.instanceCount,e.isInstancedBufferGeometry=!0,e}}class Dy extends Ti{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}class Zp{constructor(e=1,t=0,n=0){this.radius=e,this.phi=t,this.theta=n}set(e,t,n){return this.radius=e,this.phi=t,this.theta=n,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=ht(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,n){return this.radius=Math.sqrt(e*e+t*t+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,n),this.phi=Math.acos(ht(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}function Jp(i,e,t,n){const s=Py(n);switch(t){case Bg:return i*e;case Og:return i*e/s.components*s.byteLength;case Wc:return i*e/s.components*s.byteLength;case $d:return i*e*2/s.components*s.byteLength;case Zd:return i*e*2/s.components*s.byteLength;case Ug:return i*e*3/s.components*s.byteLength;case Yn:return i*e*4/s.components*s.byteLength;case Do:return i*e*4/s.components*s.byteLength;case sc:case rc:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case oc:case ac:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Of:case zf:return Math.max(i,16)*Math.max(e,8)/4;case Uf:case Nf:return Math.max(i,8)*Math.max(e,8)/2;case kf:case Hf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case Vf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Gf:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Wf:return Math.floor((i+4)/5)*Math.floor((e+3)/4)*16;case Xf:return Math.floor((i+4)/5)*Math.floor((e+4)/5)*16;case qf:return Math.floor((i+5)/6)*Math.floor((e+4)/5)*16;case Yf:return Math.floor((i+5)/6)*Math.floor((e+5)/6)*16;case Qf:return Math.floor((i+7)/8)*Math.floor((e+4)/5)*16;case Kf:return Math.floor((i+7)/8)*Math.floor((e+5)/6)*16;case jf:return Math.floor((i+7)/8)*Math.floor((e+7)/8)*16;case $f:return Math.floor((i+9)/10)*Math.floor((e+4)/5)*16;case Zf:return Math.floor((i+9)/10)*Math.floor((e+5)/6)*16;case Jf:return Math.floor((i+9)/10)*Math.floor((e+7)/8)*16;case ed:return Math.floor((i+9)/10)*Math.floor((e+9)/10)*16;case td:return Math.floor((i+11)/12)*Math.floor((e+9)/10)*16;case nd:return Math.floor((i+11)/12)*Math.floor((e+11)/12)*16;case id:case sd:case rd:return Math.ceil(i/4)*Math.ceil(e/4)*16;case od:case ad:return Math.ceil(i/4)*Math.ceil(e/4)*8;case ld:case cd:return Math.ceil(i/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function Py(i){switch(i){case ls:case Dg:return{byteLength:1,components:1};case Ya:case Pg:case Qr:return{byteLength:2,components:1};case Kd:case jd:return{byteLength:2,components:4};case Ii:case Qd:case ki:return{byteLength:4,components:1};case Fg:case Lg:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:Yd}}));typeof window<"u"&&(window.__THREE__?ut("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=Yd);function jg(){let i=null,e=!1,t=null,n=null;function s(r,o){t(r,o),n=i.requestAnimationFrame(s)}return{start:function(){e!==!0&&t!==null&&(n=i.requestAnimationFrame(s),e=!0)},stop:function(){i.cancelAnimationFrame(n),e=!1},setAnimationLoop:function(r){t=r},setContext:function(r){i=r}}}function Fy(i){const e=new WeakMap;function t(a,l){const c=a.array,u=a.usage,f=c.byteLength,d=i.createBuffer();i.bindBuffer(l,d),i.bufferData(l,c,u),a.onUploadCallback();let h;if(c instanceof Float32Array)h=i.FLOAT;else if(typeof Float16Array<"u"&&c instanceof Float16Array)h=i.HALF_FLOAT;else if(c instanceof Uint16Array)a.isFloat16BufferAttribute?h=i.HALF_FLOAT:h=i.UNSIGNED_SHORT;else if(c instanceof Int16Array)h=i.SHORT;else if(c instanceof Uint32Array)h=i.UNSIGNED_INT;else if(c instanceof Int32Array)h=i.INT;else if(c instanceof Int8Array)h=i.BYTE;else if(c instanceof Uint8Array)h=i.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)h=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:d,type:h,bytesPerElement:c.BYTES_PER_ELEMENT,version:a.version,size:f}}function n(a,l,c){const u=l.array,f=l.updateRanges;if(i.bindBuffer(c,a),f.length===0)i.bufferSubData(c,0,u);else{f.sort((h,x)=>h.start-x.start);let d=0;for(let h=1;h<f.length;h++){const x=f[d],p=f[h];p.start<=x.start+x.count+1?x.count=Math.max(x.count,p.start+p.count-x.start):(++d,f[d]=p)}f.length=d+1;for(let h=0,x=f.length;h<x;h++){const p=f[h];i.bufferSubData(c,p.start*u.BYTES_PER_ELEMENT,u,p.start,p.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(a){return a.isInterleavedBufferAttribute&&(a=a.data),e.get(a)}function r(a){a.isInterleavedBufferAttribute&&(a=a.data);const l=e.get(a);l&&(i.deleteBuffer(l.buffer),e.delete(a))}function o(a,l){if(a.isInterleavedBufferAttribute&&(a=a.data),a.isGLBufferAttribute){const u=e.get(a);(!u||u.version<a.version)&&e.set(a,{buffer:a.buffer,type:a.type,bytesPerElement:a.elementSize,version:a.version});return}const c=e.get(a);if(c===void 0)e.set(a,t(a,l));else if(c.version<a.version){if(c.size!==a.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(c.buffer,a,l),c.version=a.version}}return{get:s,remove:r,update:o}}var Ly=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,By=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,Uy=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,Oy=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Ny=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,zy=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,ky=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,Hy=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,Vy=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`,Gy=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,Wy=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Xy=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,qy=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,Yy=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,Qy=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,Ky=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,jy=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,$y=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,Zy=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Jy=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,eb=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,tb=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,nb=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`,ib=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,sb=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,rb=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,ob=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,ab=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,lb=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,cb=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,ub="gl_FragColor = linearToOutputTexel( gl_FragColor );",fb=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,db=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,hb=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,pb=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,mb=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,gb=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,xb=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,_b=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,vb=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,Ab=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,Sb=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,yb=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,bb=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,Mb=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,Cb=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,Tb=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,Eb=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,wb=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,Rb=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Ib=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,Db=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,Pb=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 uv = vec2( roughness, dotNV );
	return texture2D( dfgLUT, uv ).rg;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
vec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 dfgV = DFGApprox( vec3(0.0, 0.0, 1.0), vec3(sqrt(1.0 - dotNV * dotNV), 0.0, dotNV), material.roughness );
	vec2 dfgL = DFGApprox( vec3(0.0, 0.0, 1.0), vec3(sqrt(1.0 - dotNL * dotNL), 0.0, dotNL), material.roughness );
	vec3 FssEss_V = material.specularColor * dfgV.x + material.specularF90 * dfgV.y;
	vec3 FssEss_L = material.specularColor * dfgL.x + material.specularF90 * dfgL.y;
	float Ess_V = dfgV.x + dfgV.y;
	float Ess_L = dfgL.x + dfgL.y;
	float Ems_V = 1.0 - Ess_V;
	float Ems_L = 1.0 - Ess_L;
	vec3 Favg = material.specularColor + ( 1.0 - material.specularColor ) * 0.047619;
	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg * Favg + EPSILON );
	float compensationFactor = Ems_V * Ems_L;
	vec3 multiScatter = Fms * compensationFactor;
	return singleScatter + multiScatter;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,Fb=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,Lb=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,Bb=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,Ub=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,Ob=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Nb=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,zb=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,kb=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Hb=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Vb=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,Gb=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Wb=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Xb=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,qb=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,Yb=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Qb=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Kb=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,jb=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,$b=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,Zb=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,Jb=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,eM=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,tM=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,nM=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,iM=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,sM=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,rM=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,oM=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,aM=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,lM=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,cM=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,uM=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,fM=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,dM=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,hM=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,pM=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,mM=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		float depth = unpackRGBAToDepth( texture2D( depths, uv ) );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			return step( depth, compare );
		#else
			return step( compare, depth );
		#endif
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow( sampler2D shadow, vec2 uv, float compare ) {
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			float hard_shadow = step( distribution.x, compare );
		#else
			float hard_shadow = step( compare, distribution.x );
		#endif
		if ( hard_shadow != 1.0 ) {
			float distance = compare - distribution.x;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		
		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`,gM=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,xM=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,_M=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,vM=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,AM=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,SM=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,yM=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,bM=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,MM=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,CM=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,TM=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,EM=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,wM=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,RM=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,IM=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,DM=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,PM=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const FM=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,LM=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,BM=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,UM=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,OM=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,NM=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,zM=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,kM=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,HM=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,VM=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`,GM=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,WM=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,XM=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,qM=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,YM=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,QM=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,KM=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,jM=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,$M=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,ZM=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,JM=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,eC=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,tC=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,nC=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,iC=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,sC=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,rC=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,oC=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,aC=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,lC=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,cC=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,uC=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,fC=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,dC=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,gt={alphahash_fragment:Ly,alphahash_pars_fragment:By,alphamap_fragment:Uy,alphamap_pars_fragment:Oy,alphatest_fragment:Ny,alphatest_pars_fragment:zy,aomap_fragment:ky,aomap_pars_fragment:Hy,batching_pars_vertex:Vy,batching_vertex:Gy,begin_vertex:Wy,beginnormal_vertex:Xy,bsdfs:qy,iridescence_fragment:Yy,bumpmap_pars_fragment:Qy,clipping_planes_fragment:Ky,clipping_planes_pars_fragment:jy,clipping_planes_pars_vertex:$y,clipping_planes_vertex:Zy,color_fragment:Jy,color_pars_fragment:eb,color_pars_vertex:tb,color_vertex:nb,common:ib,cube_uv_reflection_fragment:sb,defaultnormal_vertex:rb,displacementmap_pars_vertex:ob,displacementmap_vertex:ab,emissivemap_fragment:lb,emissivemap_pars_fragment:cb,colorspace_fragment:ub,colorspace_pars_fragment:fb,envmap_fragment:db,envmap_common_pars_fragment:hb,envmap_pars_fragment:pb,envmap_pars_vertex:mb,envmap_physical_pars_fragment:Tb,envmap_vertex:gb,fog_vertex:xb,fog_pars_vertex:_b,fog_fragment:vb,fog_pars_fragment:Ab,gradientmap_pars_fragment:Sb,lightmap_pars_fragment:yb,lights_lambert_fragment:bb,lights_lambert_pars_fragment:Mb,lights_pars_begin:Cb,lights_toon_fragment:Eb,lights_toon_pars_fragment:wb,lights_phong_fragment:Rb,lights_phong_pars_fragment:Ib,lights_physical_fragment:Db,lights_physical_pars_fragment:Pb,lights_fragment_begin:Fb,lights_fragment_maps:Lb,lights_fragment_end:Bb,logdepthbuf_fragment:Ub,logdepthbuf_pars_fragment:Ob,logdepthbuf_pars_vertex:Nb,logdepthbuf_vertex:zb,map_fragment:kb,map_pars_fragment:Hb,map_particle_fragment:Vb,map_particle_pars_fragment:Gb,metalnessmap_fragment:Wb,metalnessmap_pars_fragment:Xb,morphinstance_vertex:qb,morphcolor_vertex:Yb,morphnormal_vertex:Qb,morphtarget_pars_vertex:Kb,morphtarget_vertex:jb,normal_fragment_begin:$b,normal_fragment_maps:Zb,normal_pars_fragment:Jb,normal_pars_vertex:eM,normal_vertex:tM,normalmap_pars_fragment:nM,clearcoat_normal_fragment_begin:iM,clearcoat_normal_fragment_maps:sM,clearcoat_pars_fragment:rM,iridescence_pars_fragment:oM,opaque_fragment:aM,packing:lM,premultiplied_alpha_fragment:cM,project_vertex:uM,dithering_fragment:fM,dithering_pars_fragment:dM,roughnessmap_fragment:hM,roughnessmap_pars_fragment:pM,shadowmap_pars_fragment:mM,shadowmap_pars_vertex:gM,shadowmap_vertex:xM,shadowmask_pars_fragment:_M,skinbase_vertex:vM,skinning_pars_vertex:AM,skinning_vertex:SM,skinnormal_vertex:yM,specularmap_fragment:bM,specularmap_pars_fragment:MM,tonemapping_fragment:CM,tonemapping_pars_fragment:TM,transmission_fragment:EM,transmission_pars_fragment:wM,uv_pars_fragment:RM,uv_pars_vertex:IM,uv_vertex:DM,worldpos_vertex:PM,background_vert:FM,background_frag:LM,backgroundCube_vert:BM,backgroundCube_frag:UM,cube_vert:OM,cube_frag:NM,depth_vert:zM,depth_frag:kM,distanceRGBA_vert:HM,distanceRGBA_frag:VM,equirect_vert:GM,equirect_frag:WM,linedashed_vert:XM,linedashed_frag:qM,meshbasic_vert:YM,meshbasic_frag:QM,meshlambert_vert:KM,meshlambert_frag:jM,meshmatcap_vert:$M,meshmatcap_frag:ZM,meshnormal_vert:JM,meshnormal_frag:eC,meshphong_vert:tC,meshphong_frag:nC,meshphysical_vert:iC,meshphysical_frag:sC,meshtoon_vert:rC,meshtoon_frag:oC,points_vert:aC,points_frag:lC,shadow_vert:cC,shadow_frag:uC,sprite_vert:fC,sprite_frag:dC},ke={common:{diffuse:{value:new vt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new at},alphaMap:{value:null},alphaMapTransform:{value:new at},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new at}},envmap:{envMap:{value:null},envMapRotation:{value:new at},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new at}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new at}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new at},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new at},normalScale:{value:new Ke(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new at},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new at}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new at}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new at}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new vt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new vt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new at},alphaTest:{value:0},uvTransform:{value:new at}},sprite:{diffuse:{value:new vt(16777215)},opacity:{value:1},center:{value:new Ke(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new at},alphaMap:{value:null},alphaMapTransform:{value:new at},alphaTest:{value:0}}},$i={basic:{uniforms:Vn([ke.common,ke.specularmap,ke.envmap,ke.aomap,ke.lightmap,ke.fog]),vertexShader:gt.meshbasic_vert,fragmentShader:gt.meshbasic_frag},lambert:{uniforms:Vn([ke.common,ke.specularmap,ke.envmap,ke.aomap,ke.lightmap,ke.emissivemap,ke.bumpmap,ke.normalmap,ke.displacementmap,ke.fog,ke.lights,{emissive:{value:new vt(0)}}]),vertexShader:gt.meshlambert_vert,fragmentShader:gt.meshlambert_frag},phong:{uniforms:Vn([ke.common,ke.specularmap,ke.envmap,ke.aomap,ke.lightmap,ke.emissivemap,ke.bumpmap,ke.normalmap,ke.displacementmap,ke.fog,ke.lights,{emissive:{value:new vt(0)},specular:{value:new vt(1118481)},shininess:{value:30}}]),vertexShader:gt.meshphong_vert,fragmentShader:gt.meshphong_frag},standard:{uniforms:Vn([ke.common,ke.envmap,ke.aomap,ke.lightmap,ke.emissivemap,ke.bumpmap,ke.normalmap,ke.displacementmap,ke.roughnessmap,ke.metalnessmap,ke.fog,ke.lights,{emissive:{value:new vt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:gt.meshphysical_vert,fragmentShader:gt.meshphysical_frag},toon:{uniforms:Vn([ke.common,ke.aomap,ke.lightmap,ke.emissivemap,ke.bumpmap,ke.normalmap,ke.displacementmap,ke.gradientmap,ke.fog,ke.lights,{emissive:{value:new vt(0)}}]),vertexShader:gt.meshtoon_vert,fragmentShader:gt.meshtoon_frag},matcap:{uniforms:Vn([ke.common,ke.bumpmap,ke.normalmap,ke.displacementmap,ke.fog,{matcap:{value:null}}]),vertexShader:gt.meshmatcap_vert,fragmentShader:gt.meshmatcap_frag},points:{uniforms:Vn([ke.points,ke.fog]),vertexShader:gt.points_vert,fragmentShader:gt.points_frag},dashed:{uniforms:Vn([ke.common,ke.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:gt.linedashed_vert,fragmentShader:gt.linedashed_frag},depth:{uniforms:Vn([ke.common,ke.displacementmap]),vertexShader:gt.depth_vert,fragmentShader:gt.depth_frag},normal:{uniforms:Vn([ke.common,ke.bumpmap,ke.normalmap,ke.displacementmap,{opacity:{value:1}}]),vertexShader:gt.meshnormal_vert,fragmentShader:gt.meshnormal_frag},sprite:{uniforms:Vn([ke.sprite,ke.fog]),vertexShader:gt.sprite_vert,fragmentShader:gt.sprite_frag},background:{uniforms:{uvTransform:{value:new at},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:gt.background_vert,fragmentShader:gt.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new at}},vertexShader:gt.backgroundCube_vert,fragmentShader:gt.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:gt.cube_vert,fragmentShader:gt.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:gt.equirect_vert,fragmentShader:gt.equirect_frag},distanceRGBA:{uniforms:Vn([ke.common,ke.displacementmap,{referencePosition:{value:new U},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:gt.distanceRGBA_vert,fragmentShader:gt.distanceRGBA_frag},shadow:{uniforms:Vn([ke.lights,ke.fog,{color:{value:new vt(0)},opacity:{value:1}}]),vertexShader:gt.shadow_vert,fragmentShader:gt.shadow_frag}};$i.physical={uniforms:Vn([$i.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new at},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new at},clearcoatNormalScale:{value:new Ke(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new at},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new at},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new at},sheen:{value:0},sheenColor:{value:new vt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new at},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new at},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new at},transmissionSamplerSize:{value:new Ke},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new at},attenuationDistance:{value:0},attenuationColor:{value:new vt(0)},specularColor:{value:new vt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new at},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new at},anisotropyVector:{value:new Ke},anisotropyMap:{value:null},anisotropyMapTransform:{value:new at}}]),vertexShader:gt.meshphysical_vert,fragmentShader:gt.meshphysical_frag};const Gl={r:0,b:0,g:0},Mr=new Gi,hC=new rt;function pC(i,e,t,n,s,r,o){const a=new vt(0);let l=r===!0?0:1,c,u,f=null,d=0,h=null;function x(v){let A=v.isScene===!0?v.background:null;return A&&A.isTexture&&(A=(v.backgroundBlurriness>0?t:e).get(A)),A}function p(v){let A=!1;const S=x(v);S===null?m(a,l):S&&S.isColor&&(m(S,1),A=!0);const y=i.xr.getEnvironmentBlendMode();y==="additive"?n.buffers.color.setClear(0,0,0,1,o):y==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,o),(i.autoClear||A)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function g(v,A){const S=x(A);S&&(S.isCubeTexture||S.mapping===Gc)?(u===void 0&&(u=new un(new na(1,1,1),new Kn({name:"BackgroundCubeMaterial",uniforms:qo($i.backgroundCube.uniforms),vertexShader:$i.backgroundCube.vertexShader,fragmentShader:$i.backgroundCube.fragmentShader,side:ti,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),u.geometry.deleteAttribute("normal"),u.geometry.deleteAttribute("uv"),u.onBeforeRender=function(y,M,E){this.matrixWorld.copyPosition(E.matrixWorld)},Object.defineProperty(u.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),s.update(u)),Mr.copy(A.backgroundRotation),Mr.x*=-1,Mr.y*=-1,Mr.z*=-1,S.isCubeTexture&&S.isRenderTargetTexture===!1&&(Mr.y*=-1,Mr.z*=-1),u.material.uniforms.envMap.value=S,u.material.uniforms.flipEnvMap.value=S.isCubeTexture&&S.isRenderTargetTexture===!1?-1:1,u.material.uniforms.backgroundBlurriness.value=A.backgroundBlurriness,u.material.uniforms.backgroundIntensity.value=A.backgroundIntensity,u.material.uniforms.backgroundRotation.value.setFromMatrix4(hC.makeRotationFromEuler(Mr)),u.material.toneMapped=yt.getTransfer(S.colorSpace)!==Ut,(f!==S||d!==S.version||h!==i.toneMapping)&&(u.material.needsUpdate=!0,f=S,d=S.version,h=i.toneMapping),u.layers.enableAll(),v.unshift(u,u.geometry,u.material,0,0,null)):S&&S.isTexture&&(c===void 0&&(c=new un(new Yo(2,2),new Kn({name:"BackgroundMaterial",uniforms:qo($i.background.uniforms),vertexShader:$i.background.vertexShader,fragmentShader:$i.background.fragmentShader,side:as,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),s.update(c)),c.material.uniforms.t2D.value=S,c.material.uniforms.backgroundIntensity.value=A.backgroundIntensity,c.material.toneMapped=yt.getTransfer(S.colorSpace)!==Ut,S.matrixAutoUpdate===!0&&S.updateMatrix(),c.material.uniforms.uvTransform.value.copy(S.matrix),(f!==S||d!==S.version||h!==i.toneMapping)&&(c.material.needsUpdate=!0,f=S,d=S.version,h=i.toneMapping),c.layers.enableAll(),v.unshift(c,c.geometry,c.material,0,0,null))}function m(v,A){v.getRGB(Gl,Xg(i)),n.buffers.color.setClear(Gl.r,Gl.g,Gl.b,A,o)}function _(){u!==void 0&&(u.geometry.dispose(),u.material.dispose(),u=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(v,A=1){a.set(v),l=A,m(a,l)},getClearAlpha:function(){return l},setClearAlpha:function(v){l=v,m(a,l)},render:p,addToRenderList:g,dispose:_}}function mC(i,e){const t=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},s=d(null);let r=s,o=!1;function a(C,P,B,N,k){let V=!1;const F=f(N,B,P);r!==F&&(r=F,c(r.object)),V=h(C,N,B,k),V&&x(C,N,B,k),k!==null&&e.update(k,i.ELEMENT_ARRAY_BUFFER),(V||o)&&(o=!1,A(C,P,B,N),k!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,e.get(k).buffer))}function l(){return i.createVertexArray()}function c(C){return i.bindVertexArray(C)}function u(C){return i.deleteVertexArray(C)}function f(C,P,B){const N=B.wireframe===!0;let k=n[C.id];k===void 0&&(k={},n[C.id]=k);let V=k[P.id];V===void 0&&(V={},k[P.id]=V);let F=V[N];return F===void 0&&(F=d(l()),V[N]=F),F}function d(C){const P=[],B=[],N=[];for(let k=0;k<t;k++)P[k]=0,B[k]=0,N[k]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:P,enabledAttributes:B,attributeDivisors:N,object:C,attributes:{},index:null}}function h(C,P,B,N){const k=r.attributes,V=P.attributes;let F=0;const q=B.getAttributes();for(const G in q)if(q[G].location>=0){const ce=k[G];let ye=V[G];if(ye===void 0&&(G==="instanceMatrix"&&C.instanceMatrix&&(ye=C.instanceMatrix),G==="instanceColor"&&C.instanceColor&&(ye=C.instanceColor)),ce===void 0||ce.attribute!==ye||ye&&ce.data!==ye.data)return!0;F++}return r.attributesNum!==F||r.index!==N}function x(C,P,B,N){const k={},V=P.attributes;let F=0;const q=B.getAttributes();for(const G in q)if(q[G].location>=0){let ce=V[G];ce===void 0&&(G==="instanceMatrix"&&C.instanceMatrix&&(ce=C.instanceMatrix),G==="instanceColor"&&C.instanceColor&&(ce=C.instanceColor));const ye={};ye.attribute=ce,ce&&ce.data&&(ye.data=ce.data),k[G]=ye,F++}r.attributes=k,r.attributesNum=F,r.index=N}function p(){const C=r.newAttributes;for(let P=0,B=C.length;P<B;P++)C[P]=0}function g(C){m(C,0)}function m(C,P){const B=r.newAttributes,N=r.enabledAttributes,k=r.attributeDivisors;B[C]=1,N[C]===0&&(i.enableVertexAttribArray(C),N[C]=1),k[C]!==P&&(i.vertexAttribDivisor(C,P),k[C]=P)}function _(){const C=r.newAttributes,P=r.enabledAttributes;for(let B=0,N=P.length;B<N;B++)P[B]!==C[B]&&(i.disableVertexAttribArray(B),P[B]=0)}function v(C,P,B,N,k,V,F){F===!0?i.vertexAttribIPointer(C,P,B,k,V):i.vertexAttribPointer(C,P,B,N,k,V)}function A(C,P,B,N){p();const k=N.attributes,V=B.getAttributes(),F=P.defaultAttributeValues;for(const q in V){const G=V[q];if(G.location>=0){let j=k[q];if(j===void 0&&(q==="instanceMatrix"&&C.instanceMatrix&&(j=C.instanceMatrix),q==="instanceColor"&&C.instanceColor&&(j=C.instanceColor)),j!==void 0){const ce=j.normalized,ye=j.itemSize,Ie=e.get(j);if(Ie===void 0)continue;const Ne=Ie.buffer,de=Ie.type,qe=Ie.bytesPerElement,ee=de===i.INT||de===i.UNSIGNED_INT||j.gpuType===Qd;if(j.isInterleavedBufferAttribute){const ue=j.data,Me=ue.stride,Ge=j.offset;if(ue.isInstancedInterleavedBuffer){for(let Ce=0;Ce<G.locationSize;Ce++)m(G.location+Ce,ue.meshPerAttribute);C.isInstancedMesh!==!0&&N._maxInstanceCount===void 0&&(N._maxInstanceCount=ue.meshPerAttribute*ue.count)}else for(let Ce=0;Ce<G.locationSize;Ce++)g(G.location+Ce);i.bindBuffer(i.ARRAY_BUFFER,Ne);for(let Ce=0;Ce<G.locationSize;Ce++)v(G.location+Ce,ye/G.locationSize,de,ce,Me*qe,(Ge+ye/G.locationSize*Ce)*qe,ee)}else{if(j.isInstancedBufferAttribute){for(let ue=0;ue<G.locationSize;ue++)m(G.location+ue,j.meshPerAttribute);C.isInstancedMesh!==!0&&N._maxInstanceCount===void 0&&(N._maxInstanceCount=j.meshPerAttribute*j.count)}else for(let ue=0;ue<G.locationSize;ue++)g(G.location+ue);i.bindBuffer(i.ARRAY_BUFFER,Ne);for(let ue=0;ue<G.locationSize;ue++)v(G.location+ue,ye/G.locationSize,de,ce,ye*qe,ye/G.locationSize*ue*qe,ee)}}else if(F!==void 0){const ce=F[q];if(ce!==void 0)switch(ce.length){case 2:i.vertexAttrib2fv(G.location,ce);break;case 3:i.vertexAttrib3fv(G.location,ce);break;case 4:i.vertexAttrib4fv(G.location,ce);break;default:i.vertexAttrib1fv(G.location,ce)}}}}_()}function S(){E();for(const C in n){const P=n[C];for(const B in P){const N=P[B];for(const k in N)u(N[k].object),delete N[k];delete P[B]}delete n[C]}}function y(C){if(n[C.id]===void 0)return;const P=n[C.id];for(const B in P){const N=P[B];for(const k in N)u(N[k].object),delete N[k];delete P[B]}delete n[C.id]}function M(C){for(const P in n){const B=n[P];if(B[C.id]===void 0)continue;const N=B[C.id];for(const k in N)u(N[k].object),delete N[k];delete B[C.id]}}function E(){b(),o=!0,r!==s&&(r=s,c(r.object))}function b(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:a,reset:E,resetDefaultState:b,dispose:S,releaseStatesOfGeometry:y,releaseStatesOfProgram:M,initAttributes:p,enableAttribute:g,disableUnusedAttributes:_}}function gC(i,e,t){let n;function s(c){n=c}function r(c,u){i.drawArrays(n,c,u),t.update(u,n,1)}function o(c,u,f){f!==0&&(i.drawArraysInstanced(n,c,u,f),t.update(u,n,f))}function a(c,u,f){if(f===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,c,0,u,0,f);let h=0;for(let x=0;x<f;x++)h+=u[x];t.update(h,n,1)}function l(c,u,f,d){if(f===0)return;const h=e.get("WEBGL_multi_draw");if(h===null)for(let x=0;x<c.length;x++)o(c[x],u[x],d[x]);else{h.multiDrawArraysInstancedWEBGL(n,c,0,u,0,d,0,f);let x=0;for(let p=0;p<f;p++)x+=u[p]*d[p];t.update(x,n,1)}}this.setMode=s,this.render=r,this.renderInstances=o,this.renderMultiDraw=a,this.renderMultiDrawInstances=l}function xC(i,e,t,n){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){const M=e.get("EXT_texture_filter_anisotropic");s=i.getParameter(M.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function o(M){return!(M!==Yn&&n.convert(M)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function a(M){const E=M===Qr&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(M!==ls&&n.convert(M)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&M!==ki&&!E)}function l(M){if(M==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";M="mediump"}return M==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=t.precision!==void 0?t.precision:"highp";const u=l(c);u!==c&&(ut("WebGLRenderer:",c,"not supported, using",u,"instead."),c=u);const f=t.logarithmicDepthBuffer===!0,d=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),h=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),x=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),p=i.getParameter(i.MAX_TEXTURE_SIZE),g=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),m=i.getParameter(i.MAX_VERTEX_ATTRIBS),_=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),v=i.getParameter(i.MAX_VARYING_VECTORS),A=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),S=x>0,y=i.getParameter(i.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:o,textureTypeReadable:a,precision:c,logarithmicDepthBuffer:f,reversedDepthBuffer:d,maxTextures:h,maxVertexTextures:x,maxTextureSize:p,maxCubemapSize:g,maxAttributes:m,maxVertexUniforms:_,maxVaryings:v,maxFragmentUniforms:A,vertexTextures:S,maxSamples:y}}function _C(i){const e=this;let t=null,n=0,s=!1,r=!1;const o=new Ks,a=new at,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(f,d){const h=f.length!==0||d||n!==0||s;return s=d,n=f.length,h},this.beginShadows=function(){r=!0,u(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(f,d){t=u(f,d,0)},this.setState=function(f,d,h){const x=f.clippingPlanes,p=f.clipIntersection,g=f.clipShadows,m=i.get(f);if(!s||x===null||x.length===0||r&&!g)r?u(null):c();else{const _=r?0:n,v=_*4;let A=m.clippingState||null;l.value=A,A=u(x,d,v,h);for(let S=0;S!==v;++S)A[S]=t[S];m.clippingState=A,this.numIntersection=p?this.numPlanes:0,this.numPlanes+=_}};function c(){l.value!==t&&(l.value=t,l.needsUpdate=n>0),e.numPlanes=n,e.numIntersection=0}function u(f,d,h,x){const p=f!==null?f.length:0;let g=null;if(p!==0){if(g=l.value,x!==!0||g===null){const m=h+p*4,_=d.matrixWorldInverse;a.getNormalMatrix(_),(g===null||g.length<m)&&(g=new Float32Array(m));for(let v=0,A=h;v!==p;++v,A+=4)o.copy(f[v]).applyMatrix4(_,a),o.normal.toArray(g,A),g[A+3]=o.constant}l.value=g,l.needsUpdate=!0}return e.numPlanes=p,e.numIntersection=0,g}}function vC(i){let e=new WeakMap;function t(o,a){return a===Pf?o.mapping=Vo:a===Ff&&(o.mapping=Go),o}function n(o){if(o&&o.isTexture){const a=o.mapping;if(a===Pf||a===Ff)if(e.has(o)){const l=e.get(o).texture;return t(l,o.mapping)}else{const l=o.image;if(l&&l.height>0){const c=new _y(l.height);return c.fromEquirectangularTexture(i,o),e.set(o,c),o.addEventListener("dispose",s),t(c.texture,o.mapping)}else return null}}return o}function s(o){const a=o.target;a.removeEventListener("dispose",s);const l=e.get(a);l!==void 0&&(e.delete(a),l.dispose())}function r(){e=new WeakMap}return{get:n,dispose:r}}const $s=4,em=[.125,.215,.35,.446,.526,.582],Ur=20,AC=256,ga=new rh,tm=new vt;let Vu=null,Gu=0,Wu=0,Xu=!1;const SC=new U;class nm{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,s=100,r={}){const{size:o=256,position:a=SC}=r;Vu=this._renderer.getRenderTarget(),Gu=this._renderer.getActiveCubeFace(),Wu=this._renderer.getActiveMipmapLevel(),Xu=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(o);const l=this._allocateTargets();return l.depthBuffer=!0,this._sceneToCubeUV(e,n,s,l,a),t>0&&this._blur(l,0,0,t),this._applyPMREM(l),this._cleanup(l),l}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=rm(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=sm(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(Vu,Gu,Wu),this._renderer.xr.enabled=Xu,e.scissorTest=!1,go(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===Vo||e.mapping===Go?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),Vu=this._renderer.getRenderTarget(),Gu=this._renderer.getActiveCubeFace(),Wu=this._renderer.getActiveMipmapLevel(),Xu=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:Ri,minFilter:Ri,generateMipmaps:!1,type:Qr,format:Yn,colorSpace:Xo,depthBuffer:!1},s=im(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=im(e,t,n);const{_lodMax:r}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=yC(r)),this._blurMaterial=MC(r,e,t),this._ggxMaterial=bC(r,e,t)}return s}_compileMaterial(e){const t=new un(new jn,e);this._renderer.compile(t,ga)}_sceneToCubeUV(e,t,n,s,r){const l=new Ti(90,1,t,n),c=[1,-1,1,1,1,1],u=[1,1,1,-1,-1,-1],f=this._renderer,d=f.autoClear,h=f.toneMapping;f.getClearColor(tm),f.toneMapping=sr,f.autoClear=!1,f.state.buffers.depth.getReversed()&&(f.setRenderTarget(s),f.clearDepth(),f.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new un(new na,new Yr({name:"PMREM.Background",side:ti,depthWrite:!1,depthTest:!1})));const p=this._backgroundBox,g=p.material;let m=!1;const _=e.background;_?_.isColor&&(g.color.copy(_),e.background=null,m=!0):(g.color.copy(tm),m=!0);for(let v=0;v<6;v++){const A=v%3;A===0?(l.up.set(0,c[v],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x+u[v],r.y,r.z)):A===1?(l.up.set(0,0,c[v]),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y+u[v],r.z)):(l.up.set(0,c[v],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y,r.z+u[v]));const S=this._cubeSize;go(s,A*S,v>2?S:0,S,S),f.setRenderTarget(s),m&&f.render(p,l),f.render(e,l)}f.toneMapping=h,f.autoClear=d,e.background=_}_textureToCubeUV(e,t){const n=this._renderer,s=e.mapping===Vo||e.mapping===Go;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=rm()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=sm());const r=s?this._cubemapMaterial:this._equirectMaterial,o=this._lodMeshes[0];o.material=r;const a=r.uniforms;a.envMap.value=e;const l=this._cubeSize;go(t,0,0,3*l,2*l),n.setRenderTarget(t),n.render(o,ga)}_applyPMREM(e){const t=this._renderer,n=t.autoClear;t.autoClear=!1;const s=this._lodMeshes.length;for(let r=1;r<s;r++)this._applyGGXFilter(e,r-1,r);t.autoClear=n}_applyGGXFilter(e,t,n){const s=this._renderer,r=this._pingPongRenderTarget,o=this._ggxMaterial,a=this._lodMeshes[n];a.material=o;const l=o.uniforms,c=n/(this._lodMeshes.length-1),u=t/(this._lodMeshes.length-1),f=Math.sqrt(c*c-u*u),d=.05+c*.95,h=f*d,{_lodMax:x}=this,p=this._sizeLods[n],g=3*p*(n>x-$s?n-x+$s:0),m=4*(this._cubeSize-p);l.envMap.value=e.texture,l.roughness.value=h,l.mipInt.value=x-t,go(r,g,m,3*p,2*p),s.setRenderTarget(r),s.render(a,ga),l.envMap.value=r.texture,l.roughness.value=0,l.mipInt.value=x-n,go(e,g,m,3*p,2*p),s.setRenderTarget(e),s.render(a,ga)}_blur(e,t,n,s,r){const o=this._pingPongRenderTarget;this._halfBlur(e,o,t,n,s,"latitudinal",r),this._halfBlur(o,e,n,n,s,"longitudinal",r)}_halfBlur(e,t,n,s,r,o,a){const l=this._renderer,c=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&ln("blur direction must be either latitudinal or longitudinal!");const u=3,f=this._lodMeshes[s];f.material=c;const d=c.uniforms,h=this._sizeLods[n]-1,x=isFinite(r)?Math.PI/(2*h):2*Math.PI/(2*Ur-1),p=r/x,g=isFinite(r)?1+Math.floor(u*p):Ur;g>Ur&&ut(`sigmaRadians, ${r}, is too large and will clip, as it requested ${g} samples when the maximum is set to ${Ur}`);const m=[];let _=0;for(let M=0;M<Ur;++M){const E=M/p,b=Math.exp(-E*E/2);m.push(b),M===0?_+=b:M<g&&(_+=2*b)}for(let M=0;M<m.length;M++)m[M]=m[M]/_;d.envMap.value=e.texture,d.samples.value=g,d.weights.value=m,d.latitudinal.value=o==="latitudinal",a&&(d.poleAxis.value=a);const{_lodMax:v}=this;d.dTheta.value=x,d.mipInt.value=v-n;const A=this._sizeLods[s],S=3*A*(s>v-$s?s-v+$s:0),y=4*(this._cubeSize-A);go(t,S,y,3*A,2*A),l.setRenderTarget(t),l.render(f,ga)}}function yC(i){const e=[],t=[],n=[];let s=i;const r=i-$s+1+em.length;for(let o=0;o<r;o++){const a=Math.pow(2,s);e.push(a);let l=1/a;o>i-$s?l=em[o-i+$s-1]:o===0&&(l=0),t.push(l);const c=1/(a-2),u=-c,f=1+c,d=[u,u,f,u,f,f,u,u,f,f,u,f],h=6,x=6,p=3,g=2,m=1,_=new Float32Array(p*x*h),v=new Float32Array(g*x*h),A=new Float32Array(m*x*h);for(let y=0;y<h;y++){const M=y%3*2/3-1,E=y>2?0:-1,b=[M,E,0,M+2/3,E,0,M+2/3,E+1,0,M,E,0,M+2/3,E+1,0,M,E+1,0];_.set(b,p*x*y),v.set(d,g*x*y);const C=[y,y,y,y,y,y];A.set(C,m*x*y)}const S=new jn;S.setAttribute("position",new Li(_,p)),S.setAttribute("uv",new Li(v,g)),S.setAttribute("faceIndex",new Li(A,m)),n.push(new un(S,null)),s>$s&&s--}return{lodMeshes:n,sizeLods:e,sigmas:t}}function im(i,e,t){const n=new cr(i,e,t);return n.texture.mapping=Gc,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function go(i,e,t,n,s){i.viewport.set(e,t,n,s),i.scissor.set(e,t,n,s)}function bC(i,e,t){return new Kn({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:AC,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:qc(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 3.2: Transform view direction to hemisphere configuration
				vec3 Vh = normalize(vec3(alpha * V.x, alpha * V.y, V.z));

				// Section 4.1: Orthonormal basis
				float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
				vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(Vh, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + Vh.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:Ds,depthTest:!1,depthWrite:!1})}function MC(i,e,t){const n=new Float32Array(Ur),s=new U(0,1,0);return new Kn({name:"SphericalGaussianBlur",defines:{n:Ur,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:qc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:Ds,depthTest:!1,depthWrite:!1})}function sm(){return new Kn({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:qc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:Ds,depthTest:!1,depthWrite:!1})}function rm(){return new Kn({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:qc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:Ds,depthTest:!1,depthWrite:!1})}function qc(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}function CC(i){let e=new WeakMap,t=null;function n(a){if(a&&a.isTexture){const l=a.mapping,c=l===Pf||l===Ff,u=l===Vo||l===Go;if(c||u){let f=e.get(a);const d=f!==void 0?f.texture.pmremVersion:0;if(a.isRenderTargetTexture&&a.pmremVersion!==d)return t===null&&(t=new nm(i)),f=c?t.fromEquirectangular(a,f):t.fromCubemap(a,f),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),f.texture;if(f!==void 0)return f.texture;{const h=a.image;return c&&h&&h.height>0||u&&h&&s(h)?(t===null&&(t=new nm(i)),f=c?t.fromEquirectangular(a):t.fromCubemap(a),f.texture.pmremVersion=a.pmremVersion,e.set(a,f),a.addEventListener("dispose",r),f.texture):null}}}return a}function s(a){let l=0;const c=6;for(let u=0;u<c;u++)a[u]!==void 0&&l++;return l===c}function r(a){const l=a.target;l.removeEventListener("dispose",r);const c=e.get(l);c!==void 0&&(e.delete(l),c.dispose())}function o(){e=new WeakMap,t!==null&&(t.dispose(),t=null)}return{get:n,dispose:o}}function TC(i){const e={};function t(n){if(e[n]!==void 0)return e[n];const s=i.getExtension(n);return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(n){const s=t(n);return s===null&&ja("WebGLRenderer: "+n+" extension not supported."),s}}}function EC(i,e,t,n){const s={},r=new WeakMap;function o(f){const d=f.target;d.index!==null&&e.remove(d.index);for(const x in d.attributes)e.remove(d.attributes[x]);d.removeEventListener("dispose",o),delete s[d.id];const h=r.get(d);h&&(e.remove(h),r.delete(d)),n.releaseStatesOfGeometry(d),d.isInstancedBufferGeometry===!0&&delete d._maxInstanceCount,t.memory.geometries--}function a(f,d){return s[d.id]===!0||(d.addEventListener("dispose",o),s[d.id]=!0,t.memory.geometries++),d}function l(f){const d=f.attributes;for(const h in d)e.update(d[h],i.ARRAY_BUFFER)}function c(f){const d=[],h=f.index,x=f.attributes.position;let p=0;if(h!==null){const _=h.array;p=h.version;for(let v=0,A=_.length;v<A;v+=3){const S=_[v+0],y=_[v+1],M=_[v+2];d.push(S,y,y,M,M,S)}}else if(x!==void 0){const _=x.array;p=x.version;for(let v=0,A=_.length/3-1;v<A;v+=3){const S=v+0,y=v+1,M=v+2;d.push(S,y,y,M,M,S)}}else return;const g=new(zg(d)?Wg:Gg)(d,1);g.version=p;const m=r.get(f);m&&e.remove(m),r.set(f,g)}function u(f){const d=r.get(f);if(d){const h=f.index;h!==null&&d.version<h.version&&c(f)}else c(f);return r.get(f)}return{get:a,update:l,getWireframeAttribute:u}}function wC(i,e,t){let n;function s(d){n=d}let r,o;function a(d){r=d.type,o=d.bytesPerElement}function l(d,h){i.drawElements(n,h,r,d*o),t.update(h,n,1)}function c(d,h,x){x!==0&&(i.drawElementsInstanced(n,h,r,d*o,x),t.update(h,n,x))}function u(d,h,x){if(x===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,h,0,r,d,0,x);let g=0;for(let m=0;m<x;m++)g+=h[m];t.update(g,n,1)}function f(d,h,x,p){if(x===0)return;const g=e.get("WEBGL_multi_draw");if(g===null)for(let m=0;m<d.length;m++)c(d[m]/o,h[m],p[m]);else{g.multiDrawElementsInstancedWEBGL(n,h,0,r,d,0,p,0,x);let m=0;for(let _=0;_<x;_++)m+=h[_]*p[_];t.update(m,n,1)}}this.setMode=s,this.setIndex=a,this.render=l,this.renderInstances=c,this.renderMultiDraw=u,this.renderMultiDrawInstances=f}function RC(i){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,o,a){switch(t.calls++,o){case i.TRIANGLES:t.triangles+=a*(r/3);break;case i.LINES:t.lines+=a*(r/2);break;case i.LINE_STRIP:t.lines+=a*(r-1);break;case i.LINE_LOOP:t.lines+=a*r;break;case i.POINTS:t.points+=a*r;break;default:ln("WebGLInfo: Unknown draw mode:",o);break}}function s(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:s,update:n}}function IC(i,e,t){const n=new WeakMap,s=new Zt;function r(o,a,l){const c=o.morphTargetInfluences,u=a.morphAttributes.position||a.morphAttributes.normal||a.morphAttributes.color,f=u!==void 0?u.length:0;let d=n.get(a);if(d===void 0||d.count!==f){let C=function(){E.dispose(),n.delete(a),a.removeEventListener("dispose",C)};var h=C;d!==void 0&&d.texture.dispose();const x=a.morphAttributes.position!==void 0,p=a.morphAttributes.normal!==void 0,g=a.morphAttributes.color!==void 0,m=a.morphAttributes.position||[],_=a.morphAttributes.normal||[],v=a.morphAttributes.color||[];let A=0;x===!0&&(A=1),p===!0&&(A=2),g===!0&&(A=3);let S=a.attributes.position.count*A,y=1;S>e.maxTextureSize&&(y=Math.ceil(S/e.maxTextureSize),S=e.maxTextureSize);const M=new Float32Array(S*y*4*f),E=new kg(M,S,y,f);E.type=ki,E.needsUpdate=!0;const b=A*4;for(let P=0;P<f;P++){const B=m[P],N=_[P],k=v[P],V=S*y*4*P;for(let F=0;F<B.count;F++){const q=F*b;x===!0&&(s.fromBufferAttribute(B,F),M[V+q+0]=s.x,M[V+q+1]=s.y,M[V+q+2]=s.z,M[V+q+3]=0),p===!0&&(s.fromBufferAttribute(N,F),M[V+q+4]=s.x,M[V+q+5]=s.y,M[V+q+6]=s.z,M[V+q+7]=0),g===!0&&(s.fromBufferAttribute(k,F),M[V+q+8]=s.x,M[V+q+9]=s.y,M[V+q+10]=s.z,M[V+q+11]=k.itemSize===4?s.w:1)}}d={count:f,texture:E,size:new Ke(S,y)},n.set(a,d),a.addEventListener("dispose",C)}if(o.isInstancedMesh===!0&&o.morphTexture!==null)l.getUniforms().setValue(i,"morphTexture",o.morphTexture,t);else{let x=0;for(let g=0;g<c.length;g++)x+=c[g];const p=a.morphTargetsRelative?1:1-x;l.getUniforms().setValue(i,"morphTargetBaseInfluence",p),l.getUniforms().setValue(i,"morphTargetInfluences",c)}l.getUniforms().setValue(i,"morphTargetsTexture",d.texture,t),l.getUniforms().setValue(i,"morphTargetsTextureSize",d.size)}return{update:r}}function DC(i,e,t,n){let s=new WeakMap;function r(l){const c=n.render.frame,u=l.geometry,f=e.get(l,u);if(s.get(f)!==c&&(e.update(f),s.set(f,c)),l.isInstancedMesh&&(l.hasEventListener("dispose",a)===!1&&l.addEventListener("dispose",a),s.get(l)!==c&&(t.update(l.instanceMatrix,i.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,i.ARRAY_BUFFER),s.set(l,c))),l.isSkinnedMesh){const d=l.skeleton;s.get(d)!==c&&(d.update(),s.set(d,c))}return f}function o(){s=new WeakMap}function a(l){const c=l.target;c.removeEventListener("dispose",a),t.remove(c.instanceMatrix),c.instanceColor!==null&&t.remove(c.instanceColor)}return{update:r,dispose:o}}const $g=new Qn,om=new nh(1,1),Zg=new kg,Jg=new ZS,ex=new Yg,am=[],lm=[],cm=new Float32Array(16),um=new Float32Array(9),fm=new Float32Array(4);function ia(i,e,t){const n=i[0];if(n<=0||n>0)return i;const s=e*t;let r=am[s];if(r===void 0&&(r=new Float32Array(s),am[s]=r),e!==0){n.toArray(r,0);for(let o=1,a=0;o!==e;++o)a+=t,i[o].toArray(r,a)}return r}function mn(i,e){if(i.length!==e.length)return!1;for(let t=0,n=i.length;t<n;t++)if(i[t]!==e[t])return!1;return!0}function gn(i,e){for(let t=0,n=e.length;t<n;t++)i[t]=e[t]}function Yc(i,e){let t=lm[e];t===void 0&&(t=new Int32Array(e),lm[e]=t);for(let n=0;n!==e;++n)t[n]=i.allocateTextureUnit();return t}function PC(i,e){const t=this.cache;t[0]!==e&&(i.uniform1f(this.addr,e),t[0]=e)}function FC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(mn(t,e))return;i.uniform2fv(this.addr,e),gn(t,e)}}function LC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(i.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(mn(t,e))return;i.uniform3fv(this.addr,e),gn(t,e)}}function BC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(mn(t,e))return;i.uniform4fv(this.addr,e),gn(t,e)}}function UC(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(mn(t,e))return;i.uniformMatrix2fv(this.addr,!1,e),gn(t,e)}else{if(mn(t,n))return;fm.set(n),i.uniformMatrix2fv(this.addr,!1,fm),gn(t,n)}}function OC(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(mn(t,e))return;i.uniformMatrix3fv(this.addr,!1,e),gn(t,e)}else{if(mn(t,n))return;um.set(n),i.uniformMatrix3fv(this.addr,!1,um),gn(t,n)}}function NC(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(mn(t,e))return;i.uniformMatrix4fv(this.addr,!1,e),gn(t,e)}else{if(mn(t,n))return;cm.set(n),i.uniformMatrix4fv(this.addr,!1,cm),gn(t,n)}}function zC(i,e){const t=this.cache;t[0]!==e&&(i.uniform1i(this.addr,e),t[0]=e)}function kC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(mn(t,e))return;i.uniform2iv(this.addr,e),gn(t,e)}}function HC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(mn(t,e))return;i.uniform3iv(this.addr,e),gn(t,e)}}function VC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(mn(t,e))return;i.uniform4iv(this.addr,e),gn(t,e)}}function GC(i,e){const t=this.cache;t[0]!==e&&(i.uniform1ui(this.addr,e),t[0]=e)}function WC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(mn(t,e))return;i.uniform2uiv(this.addr,e),gn(t,e)}}function XC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(mn(t,e))return;i.uniform3uiv(this.addr,e),gn(t,e)}}function qC(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(mn(t,e))return;i.uniform4uiv(this.addr,e),gn(t,e)}}function YC(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_2D_SHADOW?(om.compareFunction=Ng,r=om):r=$g,t.setTexture2D(e||r,s)}function QC(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture3D(e||Jg,s)}function KC(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTextureCube(e||ex,s)}function jC(i,e,t){const n=this.cache,s=t.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),t.setTexture2DArray(e||Zg,s)}function $C(i){switch(i){case 5126:return PC;case 35664:return FC;case 35665:return LC;case 35666:return BC;case 35674:return UC;case 35675:return OC;case 35676:return NC;case 5124:case 35670:return zC;case 35667:case 35671:return kC;case 35668:case 35672:return HC;case 35669:case 35673:return VC;case 5125:return GC;case 36294:return WC;case 36295:return XC;case 36296:return qC;case 35678:case 36198:case 36298:case 36306:case 35682:return YC;case 35679:case 36299:case 36307:return QC;case 35680:case 36300:case 36308:case 36293:return KC;case 36289:case 36303:case 36311:case 36292:return jC}}function ZC(i,e){i.uniform1fv(this.addr,e)}function JC(i,e){const t=ia(e,this.size,2);i.uniform2fv(this.addr,t)}function eT(i,e){const t=ia(e,this.size,3);i.uniform3fv(this.addr,t)}function tT(i,e){const t=ia(e,this.size,4);i.uniform4fv(this.addr,t)}function nT(i,e){const t=ia(e,this.size,4);i.uniformMatrix2fv(this.addr,!1,t)}function iT(i,e){const t=ia(e,this.size,9);i.uniformMatrix3fv(this.addr,!1,t)}function sT(i,e){const t=ia(e,this.size,16);i.uniformMatrix4fv(this.addr,!1,t)}function rT(i,e){i.uniform1iv(this.addr,e)}function oT(i,e){i.uniform2iv(this.addr,e)}function aT(i,e){i.uniform3iv(this.addr,e)}function lT(i,e){i.uniform4iv(this.addr,e)}function cT(i,e){i.uniform1uiv(this.addr,e)}function uT(i,e){i.uniform2uiv(this.addr,e)}function fT(i,e){i.uniform3uiv(this.addr,e)}function dT(i,e){i.uniform4uiv(this.addr,e)}function hT(i,e,t){const n=this.cache,s=e.length,r=Yc(t,s);mn(n,r)||(i.uniform1iv(this.addr,r),gn(n,r));for(let o=0;o!==s;++o)t.setTexture2D(e[o]||$g,r[o])}function pT(i,e,t){const n=this.cache,s=e.length,r=Yc(t,s);mn(n,r)||(i.uniform1iv(this.addr,r),gn(n,r));for(let o=0;o!==s;++o)t.setTexture3D(e[o]||Jg,r[o])}function mT(i,e,t){const n=this.cache,s=e.length,r=Yc(t,s);mn(n,r)||(i.uniform1iv(this.addr,r),gn(n,r));for(let o=0;o!==s;++o)t.setTextureCube(e[o]||ex,r[o])}function gT(i,e,t){const n=this.cache,s=e.length,r=Yc(t,s);mn(n,r)||(i.uniform1iv(this.addr,r),gn(n,r));for(let o=0;o!==s;++o)t.setTexture2DArray(e[o]||Zg,r[o])}function xT(i){switch(i){case 5126:return ZC;case 35664:return JC;case 35665:return eT;case 35666:return tT;case 35674:return nT;case 35675:return iT;case 35676:return sT;case 5124:case 35670:return rT;case 35667:case 35671:return oT;case 35668:case 35672:return aT;case 35669:case 35673:return lT;case 5125:return cT;case 36294:return uT;case 36295:return fT;case 36296:return dT;case 35678:case 36198:case 36298:case 36306:case 35682:return hT;case 35679:case 36299:case 36307:return pT;case 35680:case 36300:case 36308:case 36293:return mT;case 36289:case 36303:case 36311:case 36292:return gT}}class _T{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=$C(t.type)}}class vT{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=xT(t.type)}}class AT{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){const s=this.seq;for(let r=0,o=s.length;r!==o;++r){const a=s[r];a.setValue(e,t[a.id],n)}}}const qu=/(\w+)(\])?(\[|\.)?/g;function dm(i,e){i.seq.push(e),i.map[e.id]=e}function ST(i,e,t){const n=i.name,s=n.length;for(qu.lastIndex=0;;){const r=qu.exec(n),o=qu.lastIndex;let a=r[1];const l=r[2]==="]",c=r[3];if(l&&(a=a|0),c===void 0||c==="["&&o+2===s){dm(t,c===void 0?new _T(a,i,e):new vT(a,i,e));break}else{let f=t.map[a];f===void 0&&(f=new AT(a),dm(t,f)),t=f}}}class lc{constructor(e,t){this.seq=[],this.map={};const n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let s=0;s<n;++s){const r=e.getActiveUniform(t,s),o=e.getUniformLocation(t,r.name);ST(r,o,this)}}setValue(e,t,n,s){const r=this.map[t];r!==void 0&&r.setValue(e,n,s)}setOptional(e,t,n){const s=t[n];s!==void 0&&this.setValue(e,n,s)}static upload(e,t,n,s){for(let r=0,o=t.length;r!==o;++r){const a=t[r],l=n[a.id];l.needsUpdate!==!1&&a.setValue(e,l.value,s)}}static seqWithValue(e,t){const n=[];for(let s=0,r=e.length;s!==r;++s){const o=e[s];o.id in t&&n.push(o)}return n}}function hm(i,e,t){const n=i.createShader(e);return i.shaderSource(n,t),i.compileShader(n),n}const yT=37297;let bT=0;function MT(i,e){const t=i.split(`
`),n=[],s=Math.max(e-6,0),r=Math.min(e+6,t.length);for(let o=s;o<r;o++){const a=o+1;n.push(`${a===e?">":" "} ${a}: ${t[o]}`)}return n.join(`
`)}const pm=new at;function CT(i){yt._getMatrix(pm,yt.workingColorSpace,i);const e=`mat3( ${pm.elements.map(t=>t.toFixed(4))} )`;switch(yt.getTransfer(i)){case vc:return[e,"LinearTransferOETF"];case Ut:return[e,"sRGBTransferOETF"];default:return ut("WebGLProgram: Unsupported color space: ",i),[e,"LinearTransferOETF"]}}function mm(i,e,t){const n=i.getShaderParameter(e,i.COMPILE_STATUS),r=(i.getShaderInfoLog(e)||"").trim();if(n&&r==="")return"";const o=/ERROR: 0:(\d+)/.exec(r);if(o){const a=parseInt(o[1]);return t.toUpperCase()+`

`+r+`

`+MT(i.getShaderSource(e),a)}else return r}function TT(i,e){const t=CT(e);return[`vec4 ${i}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}function ET(i,e){let t;switch(e){case cS:t="Linear";break;case uS:t="Reinhard";break;case fS:t="Cineon";break;case dS:t="ACESFilmic";break;case pS:t="AgX";break;case mS:t="Neutral";break;case hS:t="Custom";break;default:ut("WebGLProgram: Unsupported toneMapping:",e),t="Linear"}return"vec3 "+i+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const Wl=new U;function wT(){yt.getLuminanceCoefficients(Wl);const i=Wl.x.toFixed(4),e=Wl.y.toFixed(4),t=Wl.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function RT(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Aa).join(`
`)}function IT(i){const e=[];for(const t in i){const n=i[t];n!==!1&&e.push("#define "+t+" "+n)}return e.join(`
`)}function DT(i,e){const t={},n=i.getProgramParameter(e,i.ACTIVE_ATTRIBUTES);for(let s=0;s<n;s++){const r=i.getActiveAttrib(e,s),o=r.name;let a=1;r.type===i.FLOAT_MAT2&&(a=2),r.type===i.FLOAT_MAT3&&(a=3),r.type===i.FLOAT_MAT4&&(a=4),t[o]={type:r.type,location:i.getAttribLocation(e,o),locationSize:a}}return t}function Aa(i){return i!==""}function gm(i,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function xm(i,e){return i.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const PT=/^[ \t]*#include +<([\w\d./]+)>/gm;function fd(i){return i.replace(PT,LT)}const FT=new Map;function LT(i,e){let t=gt[e];if(t===void 0){const n=FT.get(e);if(n!==void 0)t=gt[n],ut('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,n);else throw new Error("Can not resolve #include <"+e+">")}return fd(t)}const BT=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function _m(i){return i.replace(BT,UT)}function UT(i,e,t,n){let s="";for(let r=parseInt(e);r<parseInt(t);r++)s+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function vm(i){let e=`precision ${i.precision} float;
	precision ${i.precision} int;
	precision ${i.precision} sampler2D;
	precision ${i.precision} samplerCube;
	precision ${i.precision} sampler3D;
	precision ${i.precision} sampler2DArray;
	precision ${i.precision} sampler2DShadow;
	precision ${i.precision} samplerCubeShadow;
	precision ${i.precision} sampler2DArrayShadow;
	precision ${i.precision} isampler2D;
	precision ${i.precision} isampler3D;
	precision ${i.precision} isamplerCube;
	precision ${i.precision} isampler2DArray;
	precision ${i.precision} usampler2D;
	precision ${i.precision} usampler3D;
	precision ${i.precision} usamplerCube;
	precision ${i.precision} usampler2DArray;
	`;return i.precision==="highp"?e+=`
#define HIGH_PRECISION`:i.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:i.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}function OT(i){let e="SHADOWMAP_TYPE_BASIC";return i.shadowMapType===Eg?e="SHADOWMAP_TYPE_PCF":i.shadowMapType===GA?e="SHADOWMAP_TYPE_PCF_SOFT":i.shadowMapType===As&&(e="SHADOWMAP_TYPE_VSM"),e}function NT(i){let e="ENVMAP_TYPE_CUBE";if(i.envMap)switch(i.envMapMode){case Vo:case Go:e="ENVMAP_TYPE_CUBE";break;case Gc:e="ENVMAP_TYPE_CUBE_UV";break}return e}function zT(i){let e="ENVMAP_MODE_REFLECTION";return i.envMap&&i.envMapMode===Go&&(e="ENVMAP_MODE_REFRACTION"),e}function kT(i){let e="ENVMAP_BLENDING_NONE";if(i.envMap)switch(i.combine){case Rg:e="ENVMAP_BLENDING_MULTIPLY";break;case aS:e="ENVMAP_BLENDING_MIX";break;case lS:e="ENVMAP_BLENDING_ADD";break}return e}function HT(i){const e=i.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,n=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:n,maxMip:t}}function VT(i,e,t,n){const s=i.getContext(),r=t.defines;let o=t.vertexShader,a=t.fragmentShader;const l=OT(t),c=NT(t),u=zT(t),f=kT(t),d=HT(t),h=RT(t),x=IT(r),p=s.createProgram();let g,m,_=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(g=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x].filter(Aa).join(`
`),g.length>0&&(g+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x].filter(Aa).join(`
`),m.length>0&&(m+=`
`)):(g=[vm(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+u:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Aa).join(`
`),m=[vm(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,x,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+c:"",t.envMap?"#define "+u:"",t.envMap?"#define "+f:"",d?"#define CUBEUV_TEXEL_WIDTH "+d.texelWidth:"",d?"#define CUBEUV_TEXEL_HEIGHT "+d.texelHeight:"",d?"#define CUBEUV_MAX_MIP "+d.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor||t.batchingColor?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==sr?"#define TONE_MAPPING":"",t.toneMapping!==sr?gt.tonemapping_pars_fragment:"",t.toneMapping!==sr?ET("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",gt.colorspace_pars_fragment,TT("linearToOutputTexel",t.outputColorSpace),wT(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(Aa).join(`
`)),o=fd(o),o=gm(o,t),o=xm(o,t),a=fd(a),a=gm(a,t),a=xm(a,t),o=_m(o),a=_m(a),t.isRawShaderMaterial!==!0&&(_=`#version 300 es
`,g=[h,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+g,m=["#define varying in",t.glslVersion===Rp?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===Rp?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);const v=_+g+o,A=_+m+a,S=hm(s,s.VERTEX_SHADER,v),y=hm(s,s.FRAGMENT_SHADER,A);s.attachShader(p,S),s.attachShader(p,y),t.index0AttributeName!==void 0?s.bindAttribLocation(p,0,t.index0AttributeName):t.morphTargets===!0&&s.bindAttribLocation(p,0,"position"),s.linkProgram(p);function M(P){if(i.debug.checkShaderErrors){const B=s.getProgramInfoLog(p)||"",N=s.getShaderInfoLog(S)||"",k=s.getShaderInfoLog(y)||"",V=B.trim(),F=N.trim(),q=k.trim();let G=!0,j=!0;if(s.getProgramParameter(p,s.LINK_STATUS)===!1)if(G=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(s,p,S,y);else{const ce=mm(s,S,"vertex"),ye=mm(s,y,"fragment");ln("THREE.WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(p,s.VALIDATE_STATUS)+`

Material Name: `+P.name+`
Material Type: `+P.type+`

Program Info Log: `+V+`
`+ce+`
`+ye)}else V!==""?ut("WebGLProgram: Program Info Log:",V):(F===""||q==="")&&(j=!1);j&&(P.diagnostics={runnable:G,programLog:V,vertexShader:{log:F,prefix:g},fragmentShader:{log:q,prefix:m}})}s.deleteShader(S),s.deleteShader(y),E=new lc(s,p),b=DT(s,p)}let E;this.getUniforms=function(){return E===void 0&&M(this),E};let b;this.getAttributes=function(){return b===void 0&&M(this),b};let C=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return C===!1&&(C=s.getProgramParameter(p,yT)),C},this.destroy=function(){n.releaseStatesOfProgram(this),s.deleteProgram(p),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=bT++,this.cacheKey=e,this.usedTimes=1,this.program=p,this.vertexShader=S,this.fragmentShader=y,this}let GT=0;class WT{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,n=e.fragmentShader,s=this._getShaderStage(t),r=this._getShaderStage(n),o=this._getShaderCacheForMaterial(e);return o.has(s)===!1&&(o.add(s),s.usedTimes++),o.has(r)===!1&&(o.add(r),r.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const n of t)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){const t=this.shaderCache;let n=t.get(e);return n===void 0&&(n=new XT(e),t.set(e,n)),n}}class XT{constructor(e){this.id=GT++,this.code=e,this.usedTimes=0}}function qT(i,e,t,n,s,r,o){const a=new Hg,l=new WT,c=new Set,u=[],f=s.logarithmicDepthBuffer,d=s.vertexTextures;let h=s.precision;const x={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function p(b){return c.add(b),b===0?"uv":`uv${b}`}function g(b,C,P,B,N){const k=B.fog,V=N.geometry,F=b.isMeshStandardMaterial?B.environment:null,q=(b.isMeshStandardMaterial?t:e).get(b.envMap||F),G=q&&q.mapping===Gc?q.image.height:null,j=x[b.type];b.precision!==null&&(h=s.getMaxPrecision(b.precision),h!==b.precision&&ut("WebGLProgram.getParameters:",b.precision,"not supported, using",h,"instead."));const ce=V.morphAttributes.position||V.morphAttributes.normal||V.morphAttributes.color,ye=ce!==void 0?ce.length:0;let Ie=0;V.morphAttributes.position!==void 0&&(Ie=1),V.morphAttributes.normal!==void 0&&(Ie=2),V.morphAttributes.color!==void 0&&(Ie=3);let Ne,de,qe,ee;if(j){const Ct=$i[j];Ne=Ct.vertexShader,de=Ct.fragmentShader}else Ne=b.vertexShader,de=b.fragmentShader,l.update(b),qe=l.getVertexShaderID(b),ee=l.getFragmentShaderID(b);const ue=i.getRenderTarget(),Me=i.state.buffers.depth.getReversed(),Ge=N.isInstancedMesh===!0,Ce=N.isBatchedMesh===!0,Ye=!!b.map,O=!!b.matcap,z=!!q,Q=!!b.aoMap,w=!!b.lightMap,ne=!!b.bumpMap,oe=!!b.normalMap,he=!!b.displacementMap,ie=!!b.emissiveMap,pe=!!b.metalnessMap,te=!!b.roughnessMap,ve=b.anisotropy>0,R=b.clearcoat>0,T=b.dispersion>0,X=b.iridescence>0,se=b.sheen>0,fe=b.transmission>0,$=ve&&!!b.anisotropyMap,Be=R&&!!b.clearcoatMap,be=R&&!!b.clearcoatNormalMap,Xe=R&&!!b.clearcoatRoughnessMap,H=X&&!!b.iridescenceMap,J=X&&!!b.iridescenceThicknessMap,_e=se&&!!b.sheenColorMap,we=se&&!!b.sheenRoughnessMap,Le=!!b.specularMap,Re=!!b.specularColorMap,$e=!!b.specularIntensityMap,W=fe&&!!b.transmissionMap,Fe=fe&&!!b.thicknessMap,Ae=!!b.gradientMap,xe=!!b.alphaMap,me=b.alphaTest>0,ge=!!b.alphaHash,We=!!b.extensions;let et=sr;b.toneMapped&&(ue===null||ue.isXRRenderTarget===!0)&&(et=i.toneMapping);const Dt={shaderID:j,shaderType:b.type,shaderName:b.name,vertexShader:Ne,fragmentShader:de,defines:b.defines,customVertexShaderID:qe,customFragmentShaderID:ee,isRawShaderMaterial:b.isRawShaderMaterial===!0,glslVersion:b.glslVersion,precision:h,batching:Ce,batchingColor:Ce&&N._colorsTexture!==null,instancing:Ge,instancingColor:Ge&&N.instanceColor!==null,instancingMorph:Ge&&N.morphTexture!==null,supportsVertexTextures:d,outputColorSpace:ue===null?i.outputColorSpace:ue.isXRRenderTarget===!0?ue.texture.colorSpace:Xo,alphaToCoverage:!!b.alphaToCoverage,map:Ye,matcap:O,envMap:z,envMapMode:z&&q.mapping,envMapCubeUVHeight:G,aoMap:Q,lightMap:w,bumpMap:ne,normalMap:oe,displacementMap:d&&he,emissiveMap:ie,normalMapObjectSpace:oe&&b.normalMapType===AS,normalMapTangentSpace:oe&&b.normalMapType===vS,metalnessMap:pe,roughnessMap:te,anisotropy:ve,anisotropyMap:$,clearcoat:R,clearcoatMap:Be,clearcoatNormalMap:be,clearcoatRoughnessMap:Xe,dispersion:T,iridescence:X,iridescenceMap:H,iridescenceThicknessMap:J,sheen:se,sheenColorMap:_e,sheenRoughnessMap:we,specularMap:Le,specularColorMap:Re,specularIntensityMap:$e,transmission:fe,transmissionMap:W,thicknessMap:Fe,gradientMap:Ae,opaque:b.transparent===!1&&b.blending===ir&&b.alphaToCoverage===!1,alphaMap:xe,alphaTest:me,alphaHash:ge,combine:b.combine,mapUv:Ye&&p(b.map.channel),aoMapUv:Q&&p(b.aoMap.channel),lightMapUv:w&&p(b.lightMap.channel),bumpMapUv:ne&&p(b.bumpMap.channel),normalMapUv:oe&&p(b.normalMap.channel),displacementMapUv:he&&p(b.displacementMap.channel),emissiveMapUv:ie&&p(b.emissiveMap.channel),metalnessMapUv:pe&&p(b.metalnessMap.channel),roughnessMapUv:te&&p(b.roughnessMap.channel),anisotropyMapUv:$&&p(b.anisotropyMap.channel),clearcoatMapUv:Be&&p(b.clearcoatMap.channel),clearcoatNormalMapUv:be&&p(b.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Xe&&p(b.clearcoatRoughnessMap.channel),iridescenceMapUv:H&&p(b.iridescenceMap.channel),iridescenceThicknessMapUv:J&&p(b.iridescenceThicknessMap.channel),sheenColorMapUv:_e&&p(b.sheenColorMap.channel),sheenRoughnessMapUv:we&&p(b.sheenRoughnessMap.channel),specularMapUv:Le&&p(b.specularMap.channel),specularColorMapUv:Re&&p(b.specularColorMap.channel),specularIntensityMapUv:$e&&p(b.specularIntensityMap.channel),transmissionMapUv:W&&p(b.transmissionMap.channel),thicknessMapUv:Fe&&p(b.thicknessMap.channel),alphaMapUv:xe&&p(b.alphaMap.channel),vertexTangents:!!V.attributes.tangent&&(oe||ve),vertexColors:b.vertexColors,vertexAlphas:b.vertexColors===!0&&!!V.attributes.color&&V.attributes.color.itemSize===4,pointsUvs:N.isPoints===!0&&!!V.attributes.uv&&(Ye||xe),fog:!!k,useFog:b.fog===!0,fogExp2:!!k&&k.isFogExp2,flatShading:b.flatShading===!0&&b.wireframe===!1,sizeAttenuation:b.sizeAttenuation===!0,logarithmicDepthBuffer:f,reversedDepthBuffer:Me,skinning:N.isSkinnedMesh===!0,morphTargets:V.morphAttributes.position!==void 0,morphNormals:V.morphAttributes.normal!==void 0,morphColors:V.morphAttributes.color!==void 0,morphTargetsCount:ye,morphTextureStride:Ie,numDirLights:C.directional.length,numPointLights:C.point.length,numSpotLights:C.spot.length,numSpotLightMaps:C.spotLightMap.length,numRectAreaLights:C.rectArea.length,numHemiLights:C.hemi.length,numDirLightShadows:C.directionalShadowMap.length,numPointLightShadows:C.pointShadowMap.length,numSpotLightShadows:C.spotShadowMap.length,numSpotLightShadowsWithMaps:C.numSpotLightShadowsWithMaps,numLightProbes:C.numLightProbes,numClippingPlanes:o.numPlanes,numClipIntersection:o.numIntersection,dithering:b.dithering,shadowMapEnabled:i.shadowMap.enabled&&P.length>0,shadowMapType:i.shadowMap.type,toneMapping:et,decodeVideoTexture:Ye&&b.map.isVideoTexture===!0&&yt.getTransfer(b.map.colorSpace)===Ut,decodeVideoTextureEmissive:ie&&b.emissiveMap.isVideoTexture===!0&&yt.getTransfer(b.emissiveMap.colorSpace)===Ut,premultipliedAlpha:b.premultipliedAlpha,doubleSided:b.side===Ei,flipSided:b.side===ti,useDepthPacking:b.depthPacking>=0,depthPacking:b.depthPacking||0,index0AttributeName:b.index0AttributeName,extensionClipCullDistance:We&&b.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(We&&b.extensions.multiDraw===!0||Ce)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:b.customProgramCacheKey()};return Dt.vertexUv1s=c.has(1),Dt.vertexUv2s=c.has(2),Dt.vertexUv3s=c.has(3),c.clear(),Dt}function m(b){const C=[];if(b.shaderID?C.push(b.shaderID):(C.push(b.customVertexShaderID),C.push(b.customFragmentShaderID)),b.defines!==void 0)for(const P in b.defines)C.push(P),C.push(b.defines[P]);return b.isRawShaderMaterial===!1&&(_(C,b),v(C,b),C.push(i.outputColorSpace)),C.push(b.customProgramCacheKey),C.join()}function _(b,C){b.push(C.precision),b.push(C.outputColorSpace),b.push(C.envMapMode),b.push(C.envMapCubeUVHeight),b.push(C.mapUv),b.push(C.alphaMapUv),b.push(C.lightMapUv),b.push(C.aoMapUv),b.push(C.bumpMapUv),b.push(C.normalMapUv),b.push(C.displacementMapUv),b.push(C.emissiveMapUv),b.push(C.metalnessMapUv),b.push(C.roughnessMapUv),b.push(C.anisotropyMapUv),b.push(C.clearcoatMapUv),b.push(C.clearcoatNormalMapUv),b.push(C.clearcoatRoughnessMapUv),b.push(C.iridescenceMapUv),b.push(C.iridescenceThicknessMapUv),b.push(C.sheenColorMapUv),b.push(C.sheenRoughnessMapUv),b.push(C.specularMapUv),b.push(C.specularColorMapUv),b.push(C.specularIntensityMapUv),b.push(C.transmissionMapUv),b.push(C.thicknessMapUv),b.push(C.combine),b.push(C.fogExp2),b.push(C.sizeAttenuation),b.push(C.morphTargetsCount),b.push(C.morphAttributeCount),b.push(C.numDirLights),b.push(C.numPointLights),b.push(C.numSpotLights),b.push(C.numSpotLightMaps),b.push(C.numHemiLights),b.push(C.numRectAreaLights),b.push(C.numDirLightShadows),b.push(C.numPointLightShadows),b.push(C.numSpotLightShadows),b.push(C.numSpotLightShadowsWithMaps),b.push(C.numLightProbes),b.push(C.shadowMapType),b.push(C.toneMapping),b.push(C.numClippingPlanes),b.push(C.numClipIntersection),b.push(C.depthPacking)}function v(b,C){a.disableAll(),C.supportsVertexTextures&&a.enable(0),C.instancing&&a.enable(1),C.instancingColor&&a.enable(2),C.instancingMorph&&a.enable(3),C.matcap&&a.enable(4),C.envMap&&a.enable(5),C.normalMapObjectSpace&&a.enable(6),C.normalMapTangentSpace&&a.enable(7),C.clearcoat&&a.enable(8),C.iridescence&&a.enable(9),C.alphaTest&&a.enable(10),C.vertexColors&&a.enable(11),C.vertexAlphas&&a.enable(12),C.vertexUv1s&&a.enable(13),C.vertexUv2s&&a.enable(14),C.vertexUv3s&&a.enable(15),C.vertexTangents&&a.enable(16),C.anisotropy&&a.enable(17),C.alphaHash&&a.enable(18),C.batching&&a.enable(19),C.dispersion&&a.enable(20),C.batchingColor&&a.enable(21),C.gradientMap&&a.enable(22),b.push(a.mask),a.disableAll(),C.fog&&a.enable(0),C.useFog&&a.enable(1),C.flatShading&&a.enable(2),C.logarithmicDepthBuffer&&a.enable(3),C.reversedDepthBuffer&&a.enable(4),C.skinning&&a.enable(5),C.morphTargets&&a.enable(6),C.morphNormals&&a.enable(7),C.morphColors&&a.enable(8),C.premultipliedAlpha&&a.enable(9),C.shadowMapEnabled&&a.enable(10),C.doubleSided&&a.enable(11),C.flipSided&&a.enable(12),C.useDepthPacking&&a.enable(13),C.dithering&&a.enable(14),C.transmission&&a.enable(15),C.sheen&&a.enable(16),C.opaque&&a.enable(17),C.pointsUvs&&a.enable(18),C.decodeVideoTexture&&a.enable(19),C.decodeVideoTextureEmissive&&a.enable(20),C.alphaToCoverage&&a.enable(21),b.push(a.mask)}function A(b){const C=x[b.type];let P;if(C){const B=$i[C];P=py.clone(B.uniforms)}else P=b.uniforms;return P}function S(b,C){let P;for(let B=0,N=u.length;B<N;B++){const k=u[B];if(k.cacheKey===C){P=k,++P.usedTimes;break}}return P===void 0&&(P=new VT(i,C,b,r),u.push(P)),P}function y(b){if(--b.usedTimes===0){const C=u.indexOf(b);u[C]=u[u.length-1],u.pop(),b.destroy()}}function M(b){l.remove(b)}function E(){l.dispose()}return{getParameters:g,getProgramCacheKey:m,getUniforms:A,acquireProgram:S,releaseProgram:y,releaseShaderCache:M,programs:u,dispose:E}}function YT(){let i=new WeakMap;function e(o){return i.has(o)}function t(o){let a=i.get(o);return a===void 0&&(a={},i.set(o,a)),a}function n(o){i.delete(o)}function s(o,a,l){i.get(o)[a]=l}function r(){i=new WeakMap}return{has:e,get:t,remove:n,update:s,dispose:r}}function QT(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.material.id!==e.material.id?i.material.id-e.material.id:i.z!==e.z?i.z-e.z:i.id-e.id}function Am(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.z!==e.z?e.z-i.z:i.id-e.id}function Sm(){const i=[];let e=0;const t=[],n=[],s=[];function r(){e=0,t.length=0,n.length=0,s.length=0}function o(f,d,h,x,p,g){let m=i[e];return m===void 0?(m={id:f.id,object:f,geometry:d,material:h,groupOrder:x,renderOrder:f.renderOrder,z:p,group:g},i[e]=m):(m.id=f.id,m.object=f,m.geometry=d,m.material=h,m.groupOrder=x,m.renderOrder=f.renderOrder,m.z=p,m.group=g),e++,m}function a(f,d,h,x,p,g){const m=o(f,d,h,x,p,g);h.transmission>0?n.push(m):h.transparent===!0?s.push(m):t.push(m)}function l(f,d,h,x,p,g){const m=o(f,d,h,x,p,g);h.transmission>0?n.unshift(m):h.transparent===!0?s.unshift(m):t.unshift(m)}function c(f,d){t.length>1&&t.sort(f||QT),n.length>1&&n.sort(d||Am),s.length>1&&s.sort(d||Am)}function u(){for(let f=e,d=i.length;f<d;f++){const h=i[f];if(h.id===null)break;h.id=null,h.object=null,h.geometry=null,h.material=null,h.group=null}}return{opaque:t,transmissive:n,transparent:s,init:r,push:a,unshift:l,finish:u,sort:c}}function KT(){let i=new WeakMap;function e(n,s){const r=i.get(n);let o;return r===void 0?(o=new Sm,i.set(n,[o])):s>=r.length?(o=new Sm,r.push(o)):o=r[s],o}function t(){i=new WeakMap}return{get:e,dispose:t}}function jT(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new U,color:new vt};break;case"SpotLight":t={position:new U,direction:new U,color:new vt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new U,color:new vt,distance:0,decay:0};break;case"HemisphereLight":t={direction:new U,skyColor:new vt,groundColor:new vt};break;case"RectAreaLight":t={color:new vt,position:new U,halfWidth:new U,halfHeight:new U};break}return i[e.id]=t,t}}}function $T(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ke};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ke};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new Ke,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[e.id]=t,t}}}let ZT=0;function JT(i,e){return(e.castShadow?2:0)-(i.castShadow?2:0)+(e.map?1:0)-(i.map?1:0)}function e1(i){const e=new jT,t=$T(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)n.probe.push(new U);const s=new U,r=new rt,o=new rt;function a(c){let u=0,f=0,d=0;for(let b=0;b<9;b++)n.probe[b].set(0,0,0);let h=0,x=0,p=0,g=0,m=0,_=0,v=0,A=0,S=0,y=0,M=0;c.sort(JT);for(let b=0,C=c.length;b<C;b++){const P=c[b],B=P.color,N=P.intensity,k=P.distance,V=P.shadow&&P.shadow.map?P.shadow.map.texture:null;if(P.isAmbientLight)u+=B.r*N,f+=B.g*N,d+=B.b*N;else if(P.isLightProbe){for(let F=0;F<9;F++)n.probe[F].addScaledVector(P.sh.coefficients[F],N);M++}else if(P.isDirectionalLight){const F=e.get(P);if(F.color.copy(P.color).multiplyScalar(P.intensity),P.castShadow){const q=P.shadow,G=t.get(P);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,n.directionalShadow[h]=G,n.directionalShadowMap[h]=V,n.directionalShadowMatrix[h]=P.shadow.matrix,_++}n.directional[h]=F,h++}else if(P.isSpotLight){const F=e.get(P);F.position.setFromMatrixPosition(P.matrixWorld),F.color.copy(B).multiplyScalar(N),F.distance=k,F.coneCos=Math.cos(P.angle),F.penumbraCos=Math.cos(P.angle*(1-P.penumbra)),F.decay=P.decay,n.spot[p]=F;const q=P.shadow;if(P.map&&(n.spotLightMap[S]=P.map,S++,q.updateMatrices(P),P.castShadow&&y++),n.spotLightMatrix[p]=q.matrix,P.castShadow){const G=t.get(P);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,n.spotShadow[p]=G,n.spotShadowMap[p]=V,A++}p++}else if(P.isRectAreaLight){const F=e.get(P);F.color.copy(B).multiplyScalar(N),F.halfWidth.set(P.width*.5,0,0),F.halfHeight.set(0,P.height*.5,0),n.rectArea[g]=F,g++}else if(P.isPointLight){const F=e.get(P);if(F.color.copy(P.color).multiplyScalar(P.intensity),F.distance=P.distance,F.decay=P.decay,P.castShadow){const q=P.shadow,G=t.get(P);G.shadowIntensity=q.intensity,G.shadowBias=q.bias,G.shadowNormalBias=q.normalBias,G.shadowRadius=q.radius,G.shadowMapSize=q.mapSize,G.shadowCameraNear=q.camera.near,G.shadowCameraFar=q.camera.far,n.pointShadow[x]=G,n.pointShadowMap[x]=V,n.pointShadowMatrix[x]=P.shadow.matrix,v++}n.point[x]=F,x++}else if(P.isHemisphereLight){const F=e.get(P);F.skyColor.copy(P.color).multiplyScalar(N),F.groundColor.copy(P.groundColor).multiplyScalar(N),n.hemi[m]=F,m++}}g>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=ke.LTC_FLOAT_1,n.rectAreaLTC2=ke.LTC_FLOAT_2):(n.rectAreaLTC1=ke.LTC_HALF_1,n.rectAreaLTC2=ke.LTC_HALF_2)),n.ambient[0]=u,n.ambient[1]=f,n.ambient[2]=d;const E=n.hash;(E.directionalLength!==h||E.pointLength!==x||E.spotLength!==p||E.rectAreaLength!==g||E.hemiLength!==m||E.numDirectionalShadows!==_||E.numPointShadows!==v||E.numSpotShadows!==A||E.numSpotMaps!==S||E.numLightProbes!==M)&&(n.directional.length=h,n.spot.length=p,n.rectArea.length=g,n.point.length=x,n.hemi.length=m,n.directionalShadow.length=_,n.directionalShadowMap.length=_,n.pointShadow.length=v,n.pointShadowMap.length=v,n.spotShadow.length=A,n.spotShadowMap.length=A,n.directionalShadowMatrix.length=_,n.pointShadowMatrix.length=v,n.spotLightMatrix.length=A+S-y,n.spotLightMap.length=S,n.numSpotLightShadowsWithMaps=y,n.numLightProbes=M,E.directionalLength=h,E.pointLength=x,E.spotLength=p,E.rectAreaLength=g,E.hemiLength=m,E.numDirectionalShadows=_,E.numPointShadows=v,E.numSpotShadows=A,E.numSpotMaps=S,E.numLightProbes=M,n.version=ZT++)}function l(c,u){let f=0,d=0,h=0,x=0,p=0;const g=u.matrixWorldInverse;for(let m=0,_=c.length;m<_;m++){const v=c[m];if(v.isDirectionalLight){const A=n.directional[f];A.direction.setFromMatrixPosition(v.matrixWorld),s.setFromMatrixPosition(v.target.matrixWorld),A.direction.sub(s),A.direction.transformDirection(g),f++}else if(v.isSpotLight){const A=n.spot[h];A.position.setFromMatrixPosition(v.matrixWorld),A.position.applyMatrix4(g),A.direction.setFromMatrixPosition(v.matrixWorld),s.setFromMatrixPosition(v.target.matrixWorld),A.direction.sub(s),A.direction.transformDirection(g),h++}else if(v.isRectAreaLight){const A=n.rectArea[x];A.position.setFromMatrixPosition(v.matrixWorld),A.position.applyMatrix4(g),o.identity(),r.copy(v.matrixWorld),r.premultiply(g),o.extractRotation(r),A.halfWidth.set(v.width*.5,0,0),A.halfHeight.set(0,v.height*.5,0),A.halfWidth.applyMatrix4(o),A.halfHeight.applyMatrix4(o),x++}else if(v.isPointLight){const A=n.point[d];A.position.setFromMatrixPosition(v.matrixWorld),A.position.applyMatrix4(g),d++}else if(v.isHemisphereLight){const A=n.hemi[p];A.direction.setFromMatrixPosition(v.matrixWorld),A.direction.transformDirection(g),p++}}}return{setup:a,setupView:l,state:n}}function ym(i){const e=new e1(i),t=[],n=[];function s(u){c.camera=u,t.length=0,n.length=0}function r(u){t.push(u)}function o(u){n.push(u)}function a(){e.setup(t)}function l(u){e.setupView(t,u)}const c={lightsArray:t,shadowsArray:n,camera:null,lights:e,transmissionRenderTarget:{}};return{init:s,state:c,setupLights:a,setupLightsView:l,pushLight:r,pushShadow:o}}function t1(i){let e=new WeakMap;function t(s,r=0){const o=e.get(s);let a;return o===void 0?(a=new ym(i),e.set(s,[a])):r>=o.length?(a=new ym(i),o.push(a)):a=o[r],a}function n(){e=new WeakMap}return{get:t,dispose:n}}const n1=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,i1=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;function s1(i,e,t){let n=new Qg;const s=new Ke,r=new Ke,o=new Zt,a=new wy({depthPacking:_S}),l=new Ry,c={},u=t.maxTextureSize,f={[as]:ti,[ti]:as,[Ei]:Ei},d=new Kn({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new Ke},radius:{value:4}},vertexShader:n1,fragmentShader:i1}),h=d.clone();h.defines.HORIZONTAL_PASS=1;const x=new jn;x.setAttribute("position",new Li(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const p=new un(x,d),g=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Eg;let m=this.type;this.render=function(y,M,E){if(g.enabled===!1||g.autoUpdate===!1&&g.needsUpdate===!1||y.length===0)return;const b=i.getRenderTarget(),C=i.getActiveCubeFace(),P=i.getActiveMipmapLevel(),B=i.state;B.setBlending(Ds),B.buffers.depth.getReversed()===!0?B.buffers.color.setClear(0,0,0,0):B.buffers.color.setClear(1,1,1,1),B.buffers.depth.setTest(!0),B.setScissorTest(!1);const N=m!==As&&this.type===As,k=m===As&&this.type!==As;for(let V=0,F=y.length;V<F;V++){const q=y[V],G=q.shadow;if(G===void 0){ut("WebGLShadowMap:",q,"has no shadow.");continue}if(G.autoUpdate===!1&&G.needsUpdate===!1)continue;s.copy(G.mapSize);const j=G.getFrameExtents();if(s.multiply(j),r.copy(G.mapSize),(s.x>u||s.y>u)&&(s.x>u&&(r.x=Math.floor(u/j.x),s.x=r.x*j.x,G.mapSize.x=r.x),s.y>u&&(r.y=Math.floor(u/j.y),s.y=r.y*j.y,G.mapSize.y=r.y)),G.map===null||N===!0||k===!0){const ye=this.type!==As?{minFilter:xi,magFilter:xi}:{};G.map!==null&&G.map.dispose(),G.map=new cr(s.x,s.y,ye),G.map.texture.name=q.name+".shadowMap",G.camera.updateProjectionMatrix()}i.setRenderTarget(G.map),i.clear();const ce=G.getViewportCount();for(let ye=0;ye<ce;ye++){const Ie=G.getViewport(ye);o.set(r.x*Ie.x,r.y*Ie.y,r.x*Ie.z,r.y*Ie.w),B.viewport(o),G.updateMatrices(q,ye),n=G.getFrustum(),A(M,E,G.camera,q,this.type)}G.isPointLightShadow!==!0&&this.type===As&&_(G,E),G.needsUpdate=!1}m=this.type,g.needsUpdate=!1,i.setRenderTarget(b,C,P)};function _(y,M){const E=e.update(p);d.defines.VSM_SAMPLES!==y.blurSamples&&(d.defines.VSM_SAMPLES=y.blurSamples,h.defines.VSM_SAMPLES=y.blurSamples,d.needsUpdate=!0,h.needsUpdate=!0),y.mapPass===null&&(y.mapPass=new cr(s.x,s.y)),d.uniforms.shadow_pass.value=y.map.texture,d.uniforms.resolution.value=y.mapSize,d.uniforms.radius.value=y.radius,i.setRenderTarget(y.mapPass),i.clear(),i.renderBufferDirect(M,null,E,d,p,null),h.uniforms.shadow_pass.value=y.mapPass.texture,h.uniforms.resolution.value=y.mapSize,h.uniforms.radius.value=y.radius,i.setRenderTarget(y.map),i.clear(),i.renderBufferDirect(M,null,E,h,p,null)}function v(y,M,E,b){let C=null;const P=E.isPointLight===!0?y.customDistanceMaterial:y.customDepthMaterial;if(P!==void 0)C=P;else if(C=E.isPointLight===!0?l:a,i.localClippingEnabled&&M.clipShadows===!0&&Array.isArray(M.clippingPlanes)&&M.clippingPlanes.length!==0||M.displacementMap&&M.displacementScale!==0||M.alphaMap&&M.alphaTest>0||M.map&&M.alphaTest>0||M.alphaToCoverage===!0){const B=C.uuid,N=M.uuid;let k=c[B];k===void 0&&(k={},c[B]=k);let V=k[N];V===void 0&&(V=C.clone(),k[N]=V,M.addEventListener("dispose",S)),C=V}if(C.visible=M.visible,C.wireframe=M.wireframe,b===As?C.side=M.shadowSide!==null?M.shadowSide:M.side:C.side=M.shadowSide!==null?M.shadowSide:f[M.side],C.alphaMap=M.alphaMap,C.alphaTest=M.alphaToCoverage===!0?.5:M.alphaTest,C.map=M.map,C.clipShadows=M.clipShadows,C.clippingPlanes=M.clippingPlanes,C.clipIntersection=M.clipIntersection,C.displacementMap=M.displacementMap,C.displacementScale=M.displacementScale,C.displacementBias=M.displacementBias,C.wireframeLinewidth=M.wireframeLinewidth,C.linewidth=M.linewidth,E.isPointLight===!0&&C.isMeshDistanceMaterial===!0){const B=i.properties.get(C);B.light=E}return C}function A(y,M,E,b,C){if(y.visible===!1)return;if(y.layers.test(M.layers)&&(y.isMesh||y.isLine||y.isPoints)&&(y.castShadow||y.receiveShadow&&C===As)&&(!y.frustumCulled||n.intersectsObject(y))){y.modelViewMatrix.multiplyMatrices(E.matrixWorldInverse,y.matrixWorld);const N=e.update(y),k=y.material;if(Array.isArray(k)){const V=N.groups;for(let F=0,q=V.length;F<q;F++){const G=V[F],j=k[G.materialIndex];if(j&&j.visible){const ce=v(y,j,b,C);y.onBeforeShadow(i,y,M,E,N,ce,G),i.renderBufferDirect(E,null,N,ce,y,G),y.onAfterShadow(i,y,M,E,N,ce,G)}}}else if(k.visible){const V=v(y,k,b,C);y.onBeforeShadow(i,y,M,E,N,V,null),i.renderBufferDirect(E,null,N,V,y,null),y.onAfterShadow(i,y,M,E,N,V,null)}}const B=y.children;for(let N=0,k=B.length;N<k;N++)A(B[N],M,E,b,C)}function S(y){y.target.removeEventListener("dispose",S);for(const E in c){const b=c[E],C=y.target.uuid;C in b&&(b[C].dispose(),delete b[C])}}}const r1={[Cf]:Tf,[Ef]:If,[wf]:Df,[Ho]:Rf,[Tf]:Cf,[If]:Ef,[Df]:wf,[Rf]:Ho};function o1(i,e){function t(){let W=!1;const Fe=new Zt;let Ae=null;const xe=new Zt(0,0,0,0);return{setMask:function(me){Ae!==me&&!W&&(i.colorMask(me,me,me,me),Ae=me)},setLocked:function(me){W=me},setClear:function(me,ge,We,et,Dt){Dt===!0&&(me*=et,ge*=et,We*=et),Fe.set(me,ge,We,et),xe.equals(Fe)===!1&&(i.clearColor(me,ge,We,et),xe.copy(Fe))},reset:function(){W=!1,Ae=null,xe.set(-1,0,0,0)}}}function n(){let W=!1,Fe=!1,Ae=null,xe=null,me=null;return{setReversed:function(ge){if(Fe!==ge){const We=e.get("EXT_clip_control");ge?We.clipControlEXT(We.LOWER_LEFT_EXT,We.ZERO_TO_ONE_EXT):We.clipControlEXT(We.LOWER_LEFT_EXT,We.NEGATIVE_ONE_TO_ONE_EXT),Fe=ge;const et=me;me=null,this.setClear(et)}},getReversed:function(){return Fe},setTest:function(ge){ge?ue(i.DEPTH_TEST):Me(i.DEPTH_TEST)},setMask:function(ge){Ae!==ge&&!W&&(i.depthMask(ge),Ae=ge)},setFunc:function(ge){if(Fe&&(ge=r1[ge]),xe!==ge){switch(ge){case Cf:i.depthFunc(i.NEVER);break;case Tf:i.depthFunc(i.ALWAYS);break;case Ef:i.depthFunc(i.LESS);break;case Ho:i.depthFunc(i.LEQUAL);break;case wf:i.depthFunc(i.EQUAL);break;case Rf:i.depthFunc(i.GEQUAL);break;case If:i.depthFunc(i.GREATER);break;case Df:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}xe=ge}},setLocked:function(ge){W=ge},setClear:function(ge){me!==ge&&(Fe&&(ge=1-ge),i.clearDepth(ge),me=ge)},reset:function(){W=!1,Ae=null,xe=null,me=null,Fe=!1}}}function s(){let W=!1,Fe=null,Ae=null,xe=null,me=null,ge=null,We=null,et=null,Dt=null;return{setTest:function(Ct){W||(Ct?ue(i.STENCIL_TEST):Me(i.STENCIL_TEST))},setMask:function(Ct){Fe!==Ct&&!W&&(i.stencilMask(Ct),Fe=Ct)},setFunc:function(Ct,Si,Nn){(Ae!==Ct||xe!==Si||me!==Nn)&&(i.stencilFunc(Ct,Si,Nn),Ae=Ct,xe=Si,me=Nn)},setOp:function(Ct,Si,Nn){(ge!==Ct||We!==Si||et!==Nn)&&(i.stencilOp(Ct,Si,Nn),ge=Ct,We=Si,et=Nn)},setLocked:function(Ct){W=Ct},setClear:function(Ct){Dt!==Ct&&(i.clearStencil(Ct),Dt=Ct)},reset:function(){W=!1,Fe=null,Ae=null,xe=null,me=null,ge=null,We=null,et=null,Dt=null}}}const r=new t,o=new n,a=new s,l=new WeakMap,c=new WeakMap;let u={},f={},d=new WeakMap,h=[],x=null,p=!1,g=null,m=null,_=null,v=null,A=null,S=null,y=null,M=new vt(0,0,0),E=0,b=!1,C=null,P=null,B=null,N=null,k=null;const V=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let F=!1,q=0;const G=i.getParameter(i.VERSION);G.indexOf("WebGL")!==-1?(q=parseFloat(/^WebGL (\d)/.exec(G)[1]),F=q>=1):G.indexOf("OpenGL ES")!==-1&&(q=parseFloat(/^OpenGL ES (\d)/.exec(G)[1]),F=q>=2);let j=null,ce={};const ye=i.getParameter(i.SCISSOR_BOX),Ie=i.getParameter(i.VIEWPORT),Ne=new Zt().fromArray(ye),de=new Zt().fromArray(Ie);function qe(W,Fe,Ae,xe){const me=new Uint8Array(4),ge=i.createTexture();i.bindTexture(W,ge),i.texParameteri(W,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(W,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let We=0;We<Ae;We++)W===i.TEXTURE_3D||W===i.TEXTURE_2D_ARRAY?i.texImage3D(Fe,0,i.RGBA,1,1,xe,0,i.RGBA,i.UNSIGNED_BYTE,me):i.texImage2D(Fe+We,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,me);return ge}const ee={};ee[i.TEXTURE_2D]=qe(i.TEXTURE_2D,i.TEXTURE_2D,1),ee[i.TEXTURE_CUBE_MAP]=qe(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),ee[i.TEXTURE_2D_ARRAY]=qe(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),ee[i.TEXTURE_3D]=qe(i.TEXTURE_3D,i.TEXTURE_3D,1,1),r.setClear(0,0,0,1),o.setClear(1),a.setClear(0),ue(i.DEPTH_TEST),o.setFunc(Ho),ne(!1),oe(bp),ue(i.CULL_FACE),Q(Ds);function ue(W){u[W]!==!0&&(i.enable(W),u[W]=!0)}function Me(W){u[W]!==!1&&(i.disable(W),u[W]=!1)}function Ge(W,Fe){return f[W]!==Fe?(i.bindFramebuffer(W,Fe),f[W]=Fe,W===i.DRAW_FRAMEBUFFER&&(f[i.FRAMEBUFFER]=Fe),W===i.FRAMEBUFFER&&(f[i.DRAW_FRAMEBUFFER]=Fe),!0):!1}function Ce(W,Fe){let Ae=h,xe=!1;if(W){Ae=d.get(Fe),Ae===void 0&&(Ae=[],d.set(Fe,Ae));const me=W.textures;if(Ae.length!==me.length||Ae[0]!==i.COLOR_ATTACHMENT0){for(let ge=0,We=me.length;ge<We;ge++)Ae[ge]=i.COLOR_ATTACHMENT0+ge;Ae.length=me.length,xe=!0}}else Ae[0]!==i.BACK&&(Ae[0]=i.BACK,xe=!0);xe&&i.drawBuffers(Ae)}function Ye(W){return x!==W?(i.useProgram(W),x=W,!0):!1}const O={[Br]:i.FUNC_ADD,[WA]:i.FUNC_SUBTRACT,[XA]:i.FUNC_REVERSE_SUBTRACT};O[qA]=i.MIN,O[YA]=i.MAX;const z={[QA]:i.ZERO,[KA]:i.ONE,[jA]:i.SRC_COLOR,[Xa]:i.SRC_ALPHA,[nS]:i.SRC_ALPHA_SATURATE,[eS]:i.DST_COLOR,[ZA]:i.DST_ALPHA,[$A]:i.ONE_MINUS_SRC_COLOR,[qa]:i.ONE_MINUS_SRC_ALPHA,[tS]:i.ONE_MINUS_DST_COLOR,[JA]:i.ONE_MINUS_DST_ALPHA,[iS]:i.CONSTANT_COLOR,[sS]:i.ONE_MINUS_CONSTANT_COLOR,[rS]:i.CONSTANT_ALPHA,[oS]:i.ONE_MINUS_CONSTANT_ALPHA};function Q(W,Fe,Ae,xe,me,ge,We,et,Dt,Ct){if(W===Ds){p===!0&&(Me(i.BLEND),p=!1);return}if(p===!1&&(ue(i.BLEND),p=!0),W!==wg){if(W!==g||Ct!==b){if((m!==Br||A!==Br)&&(i.blendEquation(i.FUNC_ADD),m=Br,A=Br),Ct)switch(W){case ir:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Mp:i.blendFunc(i.ONE,i.ONE);break;case Cp:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case Tp:i.blendFuncSeparate(i.DST_COLOR,i.ONE_MINUS_SRC_ALPHA,i.ZERO,i.ONE);break;default:ln("WebGLState: Invalid blending: ",W);break}else switch(W){case ir:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Mp:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE,i.ONE,i.ONE);break;case Cp:ln("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case Tp:ln("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:ln("WebGLState: Invalid blending: ",W);break}_=null,v=null,S=null,y=null,M.set(0,0,0),E=0,g=W,b=Ct}return}me=me||Fe,ge=ge||Ae,We=We||xe,(Fe!==m||me!==A)&&(i.blendEquationSeparate(O[Fe],O[me]),m=Fe,A=me),(Ae!==_||xe!==v||ge!==S||We!==y)&&(i.blendFuncSeparate(z[Ae],z[xe],z[ge],z[We]),_=Ae,v=xe,S=ge,y=We),(et.equals(M)===!1||Dt!==E)&&(i.blendColor(et.r,et.g,et.b,Dt),M.copy(et),E=Dt),g=W,b=!1}function w(W,Fe){W.side===Ei?Me(i.CULL_FACE):ue(i.CULL_FACE);let Ae=W.side===ti;Fe&&(Ae=!Ae),ne(Ae),W.blending===ir&&W.transparent===!1?Q(Ds):Q(W.blending,W.blendEquation,W.blendSrc,W.blendDst,W.blendEquationAlpha,W.blendSrcAlpha,W.blendDstAlpha,W.blendColor,W.blendAlpha,W.premultipliedAlpha),o.setFunc(W.depthFunc),o.setTest(W.depthTest),o.setMask(W.depthWrite),r.setMask(W.colorWrite);const xe=W.stencilWrite;a.setTest(xe),xe&&(a.setMask(W.stencilWriteMask),a.setFunc(W.stencilFunc,W.stencilRef,W.stencilFuncMask),a.setOp(W.stencilFail,W.stencilZFail,W.stencilZPass)),ie(W.polygonOffset,W.polygonOffsetFactor,W.polygonOffsetUnits),W.alphaToCoverage===!0?ue(i.SAMPLE_ALPHA_TO_COVERAGE):Me(i.SAMPLE_ALPHA_TO_COVERAGE)}function ne(W){C!==W&&(W?i.frontFace(i.CW):i.frontFace(i.CCW),C=W)}function oe(W){W!==HA?(ue(i.CULL_FACE),W!==P&&(W===bp?i.cullFace(i.BACK):W===VA?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):Me(i.CULL_FACE),P=W}function he(W){W!==B&&(F&&i.lineWidth(W),B=W)}function ie(W,Fe,Ae){W?(ue(i.POLYGON_OFFSET_FILL),(N!==Fe||k!==Ae)&&(i.polygonOffset(Fe,Ae),N=Fe,k=Ae)):Me(i.POLYGON_OFFSET_FILL)}function pe(W){W?ue(i.SCISSOR_TEST):Me(i.SCISSOR_TEST)}function te(W){W===void 0&&(W=i.TEXTURE0+V-1),j!==W&&(i.activeTexture(W),j=W)}function ve(W,Fe,Ae){Ae===void 0&&(j===null?Ae=i.TEXTURE0+V-1:Ae=j);let xe=ce[Ae];xe===void 0&&(xe={type:void 0,texture:void 0},ce[Ae]=xe),(xe.type!==W||xe.texture!==Fe)&&(j!==Ae&&(i.activeTexture(Ae),j=Ae),i.bindTexture(W,Fe||ee[W]),xe.type=W,xe.texture=Fe)}function R(){const W=ce[j];W!==void 0&&W.type!==void 0&&(i.bindTexture(W.type,null),W.type=void 0,W.texture=void 0)}function T(){try{i.compressedTexImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function X(){try{i.compressedTexImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function se(){try{i.texSubImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function fe(){try{i.texSubImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function $(){try{i.compressedTexSubImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function Be(){try{i.compressedTexSubImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function be(){try{i.texStorage2D(...arguments)}catch(W){W("WebGLState:",W)}}function Xe(){try{i.texStorage3D(...arguments)}catch(W){W("WebGLState:",W)}}function H(){try{i.texImage2D(...arguments)}catch(W){W("WebGLState:",W)}}function J(){try{i.texImage3D(...arguments)}catch(W){W("WebGLState:",W)}}function _e(W){Ne.equals(W)===!1&&(i.scissor(W.x,W.y,W.z,W.w),Ne.copy(W))}function we(W){de.equals(W)===!1&&(i.viewport(W.x,W.y,W.z,W.w),de.copy(W))}function Le(W,Fe){let Ae=c.get(Fe);Ae===void 0&&(Ae=new WeakMap,c.set(Fe,Ae));let xe=Ae.get(W);xe===void 0&&(xe=i.getUniformBlockIndex(Fe,W.name),Ae.set(W,xe))}function Re(W,Fe){const xe=c.get(Fe).get(W);l.get(Fe)!==xe&&(i.uniformBlockBinding(Fe,xe,W.__bindingPointIndex),l.set(Fe,xe))}function $e(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),o.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),u={},j=null,ce={},f={},d=new WeakMap,h=[],x=null,p=!1,g=null,m=null,_=null,v=null,A=null,S=null,y=null,M=new vt(0,0,0),E=0,b=!1,C=null,P=null,B=null,N=null,k=null,Ne.set(0,0,i.canvas.width,i.canvas.height),de.set(0,0,i.canvas.width,i.canvas.height),r.reset(),o.reset(),a.reset()}return{buffers:{color:r,depth:o,stencil:a},enable:ue,disable:Me,bindFramebuffer:Ge,drawBuffers:Ce,useProgram:Ye,setBlending:Q,setMaterial:w,setFlipSided:ne,setCullFace:oe,setLineWidth:he,setPolygonOffset:ie,setScissorTest:pe,activeTexture:te,bindTexture:ve,unbindTexture:R,compressedTexImage2D:T,compressedTexImage3D:X,texImage2D:H,texImage3D:J,updateUBOMapping:Le,uniformBlockBinding:Re,texStorage2D:be,texStorage3D:Xe,texSubImage2D:se,texSubImage3D:fe,compressedTexSubImage2D:$,compressedTexSubImage3D:Be,scissor:_e,viewport:we,reset:$e}}function a1(i,e,t,n,s,r,o){const a=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new Ke,u=new WeakMap;let f;const d=new WeakMap;let h=!1;try{h=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function x(R,T){return h?new OffscreenCanvas(R,T):Sc("canvas")}function p(R,T,X){let se=1;const fe=ve(R);if((fe.width>X||fe.height>X)&&(se=X/Math.max(fe.width,fe.height)),se<1)if(typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&R instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&R instanceof ImageBitmap||typeof VideoFrame<"u"&&R instanceof VideoFrame){const $=Math.floor(se*fe.width),Be=Math.floor(se*fe.height);f===void 0&&(f=x($,Be));const be=T?x($,Be):f;return be.width=$,be.height=Be,be.getContext("2d").drawImage(R,0,0,$,Be),ut("WebGLRenderer: Texture has been resized from ("+fe.width+"x"+fe.height+") to ("+$+"x"+Be+")."),be}else return"data"in R&&ut("WebGLRenderer: Image in DataTexture is too big ("+fe.width+"x"+fe.height+")."),R;return R}function g(R){return R.generateMipmaps}function m(R){i.generateMipmap(R)}function _(R){return R.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:R.isWebGL3DRenderTarget?i.TEXTURE_3D:R.isWebGLArrayRenderTarget||R.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function v(R,T,X,se,fe=!1){if(R!==null){if(i[R]!==void 0)return i[R];ut("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+R+"'")}let $=T;if(T===i.RED&&(X===i.FLOAT&&($=i.R32F),X===i.HALF_FLOAT&&($=i.R16F),X===i.UNSIGNED_BYTE&&($=i.R8)),T===i.RED_INTEGER&&(X===i.UNSIGNED_BYTE&&($=i.R8UI),X===i.UNSIGNED_SHORT&&($=i.R16UI),X===i.UNSIGNED_INT&&($=i.R32UI),X===i.BYTE&&($=i.R8I),X===i.SHORT&&($=i.R16I),X===i.INT&&($=i.R32I)),T===i.RG&&(X===i.FLOAT&&($=i.RG32F),X===i.HALF_FLOAT&&($=i.RG16F),X===i.UNSIGNED_BYTE&&($=i.RG8)),T===i.RG_INTEGER&&(X===i.UNSIGNED_BYTE&&($=i.RG8UI),X===i.UNSIGNED_SHORT&&($=i.RG16UI),X===i.UNSIGNED_INT&&($=i.RG32UI),X===i.BYTE&&($=i.RG8I),X===i.SHORT&&($=i.RG16I),X===i.INT&&($=i.RG32I)),T===i.RGB_INTEGER&&(X===i.UNSIGNED_BYTE&&($=i.RGB8UI),X===i.UNSIGNED_SHORT&&($=i.RGB16UI),X===i.UNSIGNED_INT&&($=i.RGB32UI),X===i.BYTE&&($=i.RGB8I),X===i.SHORT&&($=i.RGB16I),X===i.INT&&($=i.RGB32I)),T===i.RGBA_INTEGER&&(X===i.UNSIGNED_BYTE&&($=i.RGBA8UI),X===i.UNSIGNED_SHORT&&($=i.RGBA16UI),X===i.UNSIGNED_INT&&($=i.RGBA32UI),X===i.BYTE&&($=i.RGBA8I),X===i.SHORT&&($=i.RGBA16I),X===i.INT&&($=i.RGBA32I)),T===i.RGB&&(X===i.UNSIGNED_INT_5_9_9_9_REV&&($=i.RGB9_E5),X===i.UNSIGNED_INT_10F_11F_11F_REV&&($=i.R11F_G11F_B10F)),T===i.RGBA){const Be=fe?vc:yt.getTransfer(se);X===i.FLOAT&&($=i.RGBA32F),X===i.HALF_FLOAT&&($=i.RGBA16F),X===i.UNSIGNED_BYTE&&($=Be===Ut?i.SRGB8_ALPHA8:i.RGBA8),X===i.UNSIGNED_SHORT_4_4_4_4&&($=i.RGBA4),X===i.UNSIGNED_SHORT_5_5_5_1&&($=i.RGB5_A1)}return($===i.R16F||$===i.R32F||$===i.RG16F||$===i.RG32F||$===i.RGBA16F||$===i.RGBA32F)&&e.get("EXT_color_buffer_float"),$}function A(R,T){let X;return R?T===null||T===Ii||T===Qa?X=i.DEPTH24_STENCIL8:T===ki?X=i.DEPTH32F_STENCIL8:T===Ya&&(X=i.DEPTH24_STENCIL8,ut("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):T===null||T===Ii||T===Qa?X=i.DEPTH_COMPONENT24:T===ki?X=i.DEPTH_COMPONENT32F:T===Ya&&(X=i.DEPTH_COMPONENT16),X}function S(R,T){return g(R)===!0||R.isFramebufferTexture&&R.minFilter!==xi&&R.minFilter!==Ri?Math.log2(Math.max(T.width,T.height))+1:R.mipmaps!==void 0&&R.mipmaps.length>0?R.mipmaps.length:R.isCompressedTexture&&Array.isArray(R.image)?T.mipmaps.length:1}function y(R){const T=R.target;T.removeEventListener("dispose",y),E(T),T.isVideoTexture&&u.delete(T)}function M(R){const T=R.target;T.removeEventListener("dispose",M),C(T)}function E(R){const T=n.get(R);if(T.__webglInit===void 0)return;const X=R.source,se=d.get(X);if(se){const fe=se[T.__cacheKey];fe.usedTimes--,fe.usedTimes===0&&b(R),Object.keys(se).length===0&&d.delete(X)}n.remove(R)}function b(R){const T=n.get(R);i.deleteTexture(T.__webglTexture);const X=R.source,se=d.get(X);delete se[T.__cacheKey],o.memory.textures--}function C(R){const T=n.get(R);if(R.depthTexture&&(R.depthTexture.dispose(),n.remove(R.depthTexture)),R.isWebGLCubeRenderTarget)for(let se=0;se<6;se++){if(Array.isArray(T.__webglFramebuffer[se]))for(let fe=0;fe<T.__webglFramebuffer[se].length;fe++)i.deleteFramebuffer(T.__webglFramebuffer[se][fe]);else i.deleteFramebuffer(T.__webglFramebuffer[se]);T.__webglDepthbuffer&&i.deleteRenderbuffer(T.__webglDepthbuffer[se])}else{if(Array.isArray(T.__webglFramebuffer))for(let se=0;se<T.__webglFramebuffer.length;se++)i.deleteFramebuffer(T.__webglFramebuffer[se]);else i.deleteFramebuffer(T.__webglFramebuffer);if(T.__webglDepthbuffer&&i.deleteRenderbuffer(T.__webglDepthbuffer),T.__webglMultisampledFramebuffer&&i.deleteFramebuffer(T.__webglMultisampledFramebuffer),T.__webglColorRenderbuffer)for(let se=0;se<T.__webglColorRenderbuffer.length;se++)T.__webglColorRenderbuffer[se]&&i.deleteRenderbuffer(T.__webglColorRenderbuffer[se]);T.__webglDepthRenderbuffer&&i.deleteRenderbuffer(T.__webglDepthRenderbuffer)}const X=R.textures;for(let se=0,fe=X.length;se<fe;se++){const $=n.get(X[se]);$.__webglTexture&&(i.deleteTexture($.__webglTexture),o.memory.textures--),n.remove(X[se])}n.remove(R)}let P=0;function B(){P=0}function N(){const R=P;return R>=s.maxTextures&&ut("WebGLTextures: Trying to use "+R+" texture units while this GPU supports only "+s.maxTextures),P+=1,R}function k(R){const T=[];return T.push(R.wrapS),T.push(R.wrapT),T.push(R.wrapR||0),T.push(R.magFilter),T.push(R.minFilter),T.push(R.anisotropy),T.push(R.internalFormat),T.push(R.format),T.push(R.type),T.push(R.generateMipmaps),T.push(R.premultiplyAlpha),T.push(R.flipY),T.push(R.unpackAlignment),T.push(R.colorSpace),T.join()}function V(R,T){const X=n.get(R);if(R.isVideoTexture&&pe(R),R.isRenderTargetTexture===!1&&R.isExternalTexture!==!0&&R.version>0&&X.__version!==R.version){const se=R.image;if(se===null)ut("WebGLRenderer: Texture marked for update but no image data found.");else if(se.complete===!1)ut("WebGLRenderer: Texture marked for update but image is incomplete");else{ee(X,R,T);return}}else R.isExternalTexture&&(X.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(i.TEXTURE_2D,X.__webglTexture,i.TEXTURE0+T)}function F(R,T){const X=n.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&X.__version!==R.version){ee(X,R,T);return}else R.isExternalTexture&&(X.__webglTexture=R.sourceTexture?R.sourceTexture:null);t.bindTexture(i.TEXTURE_2D_ARRAY,X.__webglTexture,i.TEXTURE0+T)}function q(R,T){const X=n.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&X.__version!==R.version){ee(X,R,T);return}t.bindTexture(i.TEXTURE_3D,X.__webglTexture,i.TEXTURE0+T)}function G(R,T){const X=n.get(R);if(R.version>0&&X.__version!==R.version){ue(X,R,T);return}t.bindTexture(i.TEXTURE_CUBE_MAP,X.__webglTexture,i.TEXTURE0+T)}const j={[Lf]:i.REPEAT,[Is]:i.CLAMP_TO_EDGE,[Bf]:i.MIRRORED_REPEAT},ce={[xi]:i.NEAREST,[gS]:i.NEAREST_MIPMAP_NEAREST,[Sl]:i.NEAREST_MIPMAP_LINEAR,[Ri]:i.LINEAR,[gu]:i.LINEAR_MIPMAP_NEAREST,[Nr]:i.LINEAR_MIPMAP_LINEAR},ye={[SS]:i.NEVER,[ES]:i.ALWAYS,[yS]:i.LESS,[Ng]:i.LEQUAL,[bS]:i.EQUAL,[TS]:i.GEQUAL,[MS]:i.GREATER,[CS]:i.NOTEQUAL};function Ie(R,T){if(T.type===ki&&e.has("OES_texture_float_linear")===!1&&(T.magFilter===Ri||T.magFilter===gu||T.magFilter===Sl||T.magFilter===Nr||T.minFilter===Ri||T.minFilter===gu||T.minFilter===Sl||T.minFilter===Nr)&&ut("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(R,i.TEXTURE_WRAP_S,j[T.wrapS]),i.texParameteri(R,i.TEXTURE_WRAP_T,j[T.wrapT]),(R===i.TEXTURE_3D||R===i.TEXTURE_2D_ARRAY)&&i.texParameteri(R,i.TEXTURE_WRAP_R,j[T.wrapR]),i.texParameteri(R,i.TEXTURE_MAG_FILTER,ce[T.magFilter]),i.texParameteri(R,i.TEXTURE_MIN_FILTER,ce[T.minFilter]),T.compareFunction&&(i.texParameteri(R,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(R,i.TEXTURE_COMPARE_FUNC,ye[T.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(T.magFilter===xi||T.minFilter!==Sl&&T.minFilter!==Nr||T.type===ki&&e.has("OES_texture_float_linear")===!1)return;if(T.anisotropy>1||n.get(T).__currentAnisotropy){const X=e.get("EXT_texture_filter_anisotropic");i.texParameterf(R,X.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(T.anisotropy,s.getMaxAnisotropy())),n.get(T).__currentAnisotropy=T.anisotropy}}}function Ne(R,T){let X=!1;R.__webglInit===void 0&&(R.__webglInit=!0,T.addEventListener("dispose",y));const se=T.source;let fe=d.get(se);fe===void 0&&(fe={},d.set(se,fe));const $=k(T);if($!==R.__cacheKey){fe[$]===void 0&&(fe[$]={texture:i.createTexture(),usedTimes:0},o.memory.textures++,X=!0),fe[$].usedTimes++;const Be=fe[R.__cacheKey];Be!==void 0&&(fe[R.__cacheKey].usedTimes--,Be.usedTimes===0&&b(T)),R.__cacheKey=$,R.__webglTexture=fe[$].texture}return X}function de(R,T,X){return Math.floor(Math.floor(R/X)/T)}function qe(R,T,X,se){const $=R.updateRanges;if($.length===0)t.texSubImage2D(i.TEXTURE_2D,0,0,0,T.width,T.height,X,se,T.data);else{$.sort((J,_e)=>J.start-_e.start);let Be=0;for(let J=1;J<$.length;J++){const _e=$[Be],we=$[J],Le=_e.start+_e.count,Re=de(we.start,T.width,4),$e=de(_e.start,T.width,4);we.start<=Le+1&&Re===$e&&de(we.start+we.count-1,T.width,4)===Re?_e.count=Math.max(_e.count,we.start+we.count-_e.start):(++Be,$[Be]=we)}$.length=Be+1;const be=i.getParameter(i.UNPACK_ROW_LENGTH),Xe=i.getParameter(i.UNPACK_SKIP_PIXELS),H=i.getParameter(i.UNPACK_SKIP_ROWS);i.pixelStorei(i.UNPACK_ROW_LENGTH,T.width);for(let J=0,_e=$.length;J<_e;J++){const we=$[J],Le=Math.floor(we.start/4),Re=Math.ceil(we.count/4),$e=Le%T.width,W=Math.floor(Le/T.width),Fe=Re,Ae=1;i.pixelStorei(i.UNPACK_SKIP_PIXELS,$e),i.pixelStorei(i.UNPACK_SKIP_ROWS,W),t.texSubImage2D(i.TEXTURE_2D,0,$e,W,Fe,Ae,X,se,T.data)}R.clearUpdateRanges(),i.pixelStorei(i.UNPACK_ROW_LENGTH,be),i.pixelStorei(i.UNPACK_SKIP_PIXELS,Xe),i.pixelStorei(i.UNPACK_SKIP_ROWS,H)}}function ee(R,T,X){let se=i.TEXTURE_2D;(T.isDataArrayTexture||T.isCompressedArrayTexture)&&(se=i.TEXTURE_2D_ARRAY),T.isData3DTexture&&(se=i.TEXTURE_3D);const fe=Ne(R,T),$=T.source;t.bindTexture(se,R.__webglTexture,i.TEXTURE0+X);const Be=n.get($);if($.version!==Be.__version||fe===!0){t.activeTexture(i.TEXTURE0+X);const be=yt.getPrimaries(yt.workingColorSpace),Xe=T.colorSpace===js?null:yt.getPrimaries(T.colorSpace),H=T.colorSpace===js||be===Xe?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,T.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,T.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,H);let J=p(T.image,!1,s.maxTextureSize);J=te(T,J);const _e=r.convert(T.format,T.colorSpace),we=r.convert(T.type);let Le=v(T.internalFormat,_e,we,T.colorSpace,T.isVideoTexture);Ie(se,T);let Re;const $e=T.mipmaps,W=T.isVideoTexture!==!0,Fe=Be.__version===void 0||fe===!0,Ae=$.dataReady,xe=S(T,J);if(T.isDepthTexture)Le=A(T.format===Ka,T.type),Fe&&(W?t.texStorage2D(i.TEXTURE_2D,1,Le,J.width,J.height):t.texImage2D(i.TEXTURE_2D,0,Le,J.width,J.height,0,_e,we,null));else if(T.isDataTexture)if($e.length>0){W&&Fe&&t.texStorage2D(i.TEXTURE_2D,xe,Le,$e[0].width,$e[0].height);for(let me=0,ge=$e.length;me<ge;me++)Re=$e[me],W?Ae&&t.texSubImage2D(i.TEXTURE_2D,me,0,0,Re.width,Re.height,_e,we,Re.data):t.texImage2D(i.TEXTURE_2D,me,Le,Re.width,Re.height,0,_e,we,Re.data);T.generateMipmaps=!1}else W?(Fe&&t.texStorage2D(i.TEXTURE_2D,xe,Le,J.width,J.height),Ae&&qe(T,J,_e,we)):t.texImage2D(i.TEXTURE_2D,0,Le,J.width,J.height,0,_e,we,J.data);else if(T.isCompressedTexture)if(T.isCompressedArrayTexture){W&&Fe&&t.texStorage3D(i.TEXTURE_2D_ARRAY,xe,Le,$e[0].width,$e[0].height,J.depth);for(let me=0,ge=$e.length;me<ge;me++)if(Re=$e[me],T.format!==Yn)if(_e!==null)if(W){if(Ae)if(T.layerUpdates.size>0){const We=Jp(Re.width,Re.height,T.format,T.type);for(const et of T.layerUpdates){const Dt=Re.data.subarray(et*We/Re.data.BYTES_PER_ELEMENT,(et+1)*We/Re.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,me,0,0,et,Re.width,Re.height,1,_e,Dt)}T.clearLayerUpdates()}else t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,me,0,0,0,Re.width,Re.height,J.depth,_e,Re.data)}else t.compressedTexImage3D(i.TEXTURE_2D_ARRAY,me,Le,Re.width,Re.height,J.depth,0,Re.data,0,0);else ut("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else W?Ae&&t.texSubImage3D(i.TEXTURE_2D_ARRAY,me,0,0,0,Re.width,Re.height,J.depth,_e,we,Re.data):t.texImage3D(i.TEXTURE_2D_ARRAY,me,Le,Re.width,Re.height,J.depth,0,_e,we,Re.data)}else{W&&Fe&&t.texStorage2D(i.TEXTURE_2D,xe,Le,$e[0].width,$e[0].height);for(let me=0,ge=$e.length;me<ge;me++)Re=$e[me],T.format!==Yn?_e!==null?W?Ae&&t.compressedTexSubImage2D(i.TEXTURE_2D,me,0,0,Re.width,Re.height,_e,Re.data):t.compressedTexImage2D(i.TEXTURE_2D,me,Le,Re.width,Re.height,0,Re.data):ut("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):W?Ae&&t.texSubImage2D(i.TEXTURE_2D,me,0,0,Re.width,Re.height,_e,we,Re.data):t.texImage2D(i.TEXTURE_2D,me,Le,Re.width,Re.height,0,_e,we,Re.data)}else if(T.isDataArrayTexture)if(W){if(Fe&&t.texStorage3D(i.TEXTURE_2D_ARRAY,xe,Le,J.width,J.height,J.depth),Ae)if(T.layerUpdates.size>0){const me=Jp(J.width,J.height,T.format,T.type);for(const ge of T.layerUpdates){const We=J.data.subarray(ge*me/J.data.BYTES_PER_ELEMENT,(ge+1)*me/J.data.BYTES_PER_ELEMENT);t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,ge,J.width,J.height,1,_e,we,We)}T.clearLayerUpdates()}else t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,J.width,J.height,J.depth,_e,we,J.data)}else t.texImage3D(i.TEXTURE_2D_ARRAY,0,Le,J.width,J.height,J.depth,0,_e,we,J.data);else if(T.isData3DTexture)W?(Fe&&t.texStorage3D(i.TEXTURE_3D,xe,Le,J.width,J.height,J.depth),Ae&&t.texSubImage3D(i.TEXTURE_3D,0,0,0,0,J.width,J.height,J.depth,_e,we,J.data)):t.texImage3D(i.TEXTURE_3D,0,Le,J.width,J.height,J.depth,0,_e,we,J.data);else if(T.isFramebufferTexture){if(Fe)if(W)t.texStorage2D(i.TEXTURE_2D,xe,Le,J.width,J.height);else{let me=J.width,ge=J.height;for(let We=0;We<xe;We++)t.texImage2D(i.TEXTURE_2D,We,Le,me,ge,0,_e,we,null),me>>=1,ge>>=1}}else if($e.length>0){if(W&&Fe){const me=ve($e[0]);t.texStorage2D(i.TEXTURE_2D,xe,Le,me.width,me.height)}for(let me=0,ge=$e.length;me<ge;me++)Re=$e[me],W?Ae&&t.texSubImage2D(i.TEXTURE_2D,me,0,0,_e,we,Re):t.texImage2D(i.TEXTURE_2D,me,Le,_e,we,Re);T.generateMipmaps=!1}else if(W){if(Fe){const me=ve(J);t.texStorage2D(i.TEXTURE_2D,xe,Le,me.width,me.height)}Ae&&t.texSubImage2D(i.TEXTURE_2D,0,0,0,_e,we,J)}else t.texImage2D(i.TEXTURE_2D,0,Le,_e,we,J);g(T)&&m(se),Be.__version=$.version,T.onUpdate&&T.onUpdate(T)}R.__version=T.version}function ue(R,T,X){if(T.image.length!==6)return;const se=Ne(R,T),fe=T.source;t.bindTexture(i.TEXTURE_CUBE_MAP,R.__webglTexture,i.TEXTURE0+X);const $=n.get(fe);if(fe.version!==$.__version||se===!0){t.activeTexture(i.TEXTURE0+X);const Be=yt.getPrimaries(yt.workingColorSpace),be=T.colorSpace===js?null:yt.getPrimaries(T.colorSpace),Xe=T.colorSpace===js||Be===be?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,T.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,T.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,T.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,Xe);const H=T.isCompressedTexture||T.image[0].isCompressedTexture,J=T.image[0]&&T.image[0].isDataTexture,_e=[];for(let ge=0;ge<6;ge++)!H&&!J?_e[ge]=p(T.image[ge],!0,s.maxCubemapSize):_e[ge]=J?T.image[ge].image:T.image[ge],_e[ge]=te(T,_e[ge]);const we=_e[0],Le=r.convert(T.format,T.colorSpace),Re=r.convert(T.type),$e=v(T.internalFormat,Le,Re,T.colorSpace),W=T.isVideoTexture!==!0,Fe=$.__version===void 0||se===!0,Ae=fe.dataReady;let xe=S(T,we);Ie(i.TEXTURE_CUBE_MAP,T);let me;if(H){W&&Fe&&t.texStorage2D(i.TEXTURE_CUBE_MAP,xe,$e,we.width,we.height);for(let ge=0;ge<6;ge++){me=_e[ge].mipmaps;for(let We=0;We<me.length;We++){const et=me[We];T.format!==Yn?Le!==null?W?Ae&&t.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We,0,0,et.width,et.height,Le,et.data):t.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We,$e,et.width,et.height,0,et.data):ut("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):W?Ae&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We,0,0,et.width,et.height,Le,Re,et.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We,$e,et.width,et.height,0,Le,Re,et.data)}}}else{if(me=T.mipmaps,W&&Fe){me.length>0&&xe++;const ge=ve(_e[0]);t.texStorage2D(i.TEXTURE_CUBE_MAP,xe,$e,ge.width,ge.height)}for(let ge=0;ge<6;ge++)if(J){W?Ae&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,0,0,_e[ge].width,_e[ge].height,Le,Re,_e[ge].data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,$e,_e[ge].width,_e[ge].height,0,Le,Re,_e[ge].data);for(let We=0;We<me.length;We++){const Dt=me[We].image[ge].image;W?Ae&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We+1,0,0,Dt.width,Dt.height,Le,Re,Dt.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We+1,$e,Dt.width,Dt.height,0,Le,Re,Dt.data)}}else{W?Ae&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,0,0,Le,Re,_e[ge]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,0,$e,Le,Re,_e[ge]);for(let We=0;We<me.length;We++){const et=me[We];W?Ae&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We+1,0,0,Le,Re,et.image[ge]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+ge,We+1,$e,Le,Re,et.image[ge])}}}g(T)&&m(i.TEXTURE_CUBE_MAP),$.__version=fe.version,T.onUpdate&&T.onUpdate(T)}R.__version=T.version}function Me(R,T,X,se,fe,$){const Be=r.convert(X.format,X.colorSpace),be=r.convert(X.type),Xe=v(X.internalFormat,Be,be,X.colorSpace),H=n.get(T),J=n.get(X);if(J.__renderTarget=T,!H.__hasExternalTextures){const _e=Math.max(1,T.width>>$),we=Math.max(1,T.height>>$);fe===i.TEXTURE_3D||fe===i.TEXTURE_2D_ARRAY?t.texImage3D(fe,$,Xe,_e,we,T.depth,0,Be,be,null):t.texImage2D(fe,$,Xe,_e,we,0,Be,be,null)}t.bindFramebuffer(i.FRAMEBUFFER,R),ie(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,se,fe,J.__webglTexture,0,he(T)):(fe===i.TEXTURE_2D||fe>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&fe<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,se,fe,J.__webglTexture,$),t.bindFramebuffer(i.FRAMEBUFFER,null)}function Ge(R,T,X){if(i.bindRenderbuffer(i.RENDERBUFFER,R),T.depthBuffer){const se=T.depthTexture,fe=se&&se.isDepthTexture?se.type:null,$=A(T.stencilBuffer,fe),Be=T.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,be=he(T);ie(T)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,be,$,T.width,T.height):X?i.renderbufferStorageMultisample(i.RENDERBUFFER,be,$,T.width,T.height):i.renderbufferStorage(i.RENDERBUFFER,$,T.width,T.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,Be,i.RENDERBUFFER,R)}else{const se=T.textures;for(let fe=0;fe<se.length;fe++){const $=se[fe],Be=r.convert($.format,$.colorSpace),be=r.convert($.type),Xe=v($.internalFormat,Be,be,$.colorSpace),H=he(T);X&&ie(T)===!1?i.renderbufferStorageMultisample(i.RENDERBUFFER,H,Xe,T.width,T.height):ie(T)?a.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,H,Xe,T.width,T.height):i.renderbufferStorage(i.RENDERBUFFER,Xe,T.width,T.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function Ce(R,T){if(T&&T.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(t.bindFramebuffer(i.FRAMEBUFFER,R),!(T.depthTexture&&T.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const se=n.get(T.depthTexture);se.__renderTarget=T,(!se.__webglTexture||T.depthTexture.image.width!==T.width||T.depthTexture.image.height!==T.height)&&(T.depthTexture.image.width=T.width,T.depthTexture.image.height=T.height,T.depthTexture.needsUpdate=!0),V(T.depthTexture,0);const fe=se.__webglTexture,$=he(T);if(T.depthTexture.format===Wo)ie(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,fe,0,$):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,fe,0);else if(T.depthTexture.format===Ka)ie(T)?a.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,fe,0,$):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,fe,0);else throw new Error("Unknown depthTexture format")}function Ye(R){const T=n.get(R),X=R.isWebGLCubeRenderTarget===!0;if(T.__boundDepthTexture!==R.depthTexture){const se=R.depthTexture;if(T.__depthDisposeCallback&&T.__depthDisposeCallback(),se){const fe=()=>{delete T.__boundDepthTexture,delete T.__depthDisposeCallback,se.removeEventListener("dispose",fe)};se.addEventListener("dispose",fe),T.__depthDisposeCallback=fe}T.__boundDepthTexture=se}if(R.depthTexture&&!T.__autoAllocateDepthBuffer){if(X)throw new Error("target.depthTexture not supported in Cube render targets");const se=R.texture.mipmaps;se&&se.length>0?Ce(T.__webglFramebuffer[0],R):Ce(T.__webglFramebuffer,R)}else if(X){T.__webglDepthbuffer=[];for(let se=0;se<6;se++)if(t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer[se]),T.__webglDepthbuffer[se]===void 0)T.__webglDepthbuffer[se]=i.createRenderbuffer(),Ge(T.__webglDepthbuffer[se],R,!1);else{const fe=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,$=T.__webglDepthbuffer[se];i.bindRenderbuffer(i.RENDERBUFFER,$),i.framebufferRenderbuffer(i.FRAMEBUFFER,fe,i.RENDERBUFFER,$)}}else{const se=R.texture.mipmaps;if(se&&se.length>0?t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer[0]):t.bindFramebuffer(i.FRAMEBUFFER,T.__webglFramebuffer),T.__webglDepthbuffer===void 0)T.__webglDepthbuffer=i.createRenderbuffer(),Ge(T.__webglDepthbuffer,R,!1);else{const fe=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,$=T.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,$),i.framebufferRenderbuffer(i.FRAMEBUFFER,fe,i.RENDERBUFFER,$)}}t.bindFramebuffer(i.FRAMEBUFFER,null)}function O(R,T,X){const se=n.get(R);T!==void 0&&Me(se.__webglFramebuffer,R,R.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),X!==void 0&&Ye(R)}function z(R){const T=R.texture,X=n.get(R),se=n.get(T);R.addEventListener("dispose",M);const fe=R.textures,$=R.isWebGLCubeRenderTarget===!0,Be=fe.length>1;if(Be||(se.__webglTexture===void 0&&(se.__webglTexture=i.createTexture()),se.__version=T.version,o.memory.textures++),$){X.__webglFramebuffer=[];for(let be=0;be<6;be++)if(T.mipmaps&&T.mipmaps.length>0){X.__webglFramebuffer[be]=[];for(let Xe=0;Xe<T.mipmaps.length;Xe++)X.__webglFramebuffer[be][Xe]=i.createFramebuffer()}else X.__webglFramebuffer[be]=i.createFramebuffer()}else{if(T.mipmaps&&T.mipmaps.length>0){X.__webglFramebuffer=[];for(let be=0;be<T.mipmaps.length;be++)X.__webglFramebuffer[be]=i.createFramebuffer()}else X.__webglFramebuffer=i.createFramebuffer();if(Be)for(let be=0,Xe=fe.length;be<Xe;be++){const H=n.get(fe[be]);H.__webglTexture===void 0&&(H.__webglTexture=i.createTexture(),o.memory.textures++)}if(R.samples>0&&ie(R)===!1){X.__webglMultisampledFramebuffer=i.createFramebuffer(),X.__webglColorRenderbuffer=[],t.bindFramebuffer(i.FRAMEBUFFER,X.__webglMultisampledFramebuffer);for(let be=0;be<fe.length;be++){const Xe=fe[be];X.__webglColorRenderbuffer[be]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,X.__webglColorRenderbuffer[be]);const H=r.convert(Xe.format,Xe.colorSpace),J=r.convert(Xe.type),_e=v(Xe.internalFormat,H,J,Xe.colorSpace,R.isXRRenderTarget===!0),we=he(R);i.renderbufferStorageMultisample(i.RENDERBUFFER,we,_e,R.width,R.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+be,i.RENDERBUFFER,X.__webglColorRenderbuffer[be])}i.bindRenderbuffer(i.RENDERBUFFER,null),R.depthBuffer&&(X.__webglDepthRenderbuffer=i.createRenderbuffer(),Ge(X.__webglDepthRenderbuffer,R,!0)),t.bindFramebuffer(i.FRAMEBUFFER,null)}}if($){t.bindTexture(i.TEXTURE_CUBE_MAP,se.__webglTexture),Ie(i.TEXTURE_CUBE_MAP,T);for(let be=0;be<6;be++)if(T.mipmaps&&T.mipmaps.length>0)for(let Xe=0;Xe<T.mipmaps.length;Xe++)Me(X.__webglFramebuffer[be][Xe],R,T,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+be,Xe);else Me(X.__webglFramebuffer[be],R,T,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+be,0);g(T)&&m(i.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(Be){for(let be=0,Xe=fe.length;be<Xe;be++){const H=fe[be],J=n.get(H);let _e=i.TEXTURE_2D;(R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(_e=R.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(_e,J.__webglTexture),Ie(_e,H),Me(X.__webglFramebuffer,R,H,i.COLOR_ATTACHMENT0+be,_e,0),g(H)&&m(_e)}t.unbindTexture()}else{let be=i.TEXTURE_2D;if((R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(be=R.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(be,se.__webglTexture),Ie(be,T),T.mipmaps&&T.mipmaps.length>0)for(let Xe=0;Xe<T.mipmaps.length;Xe++)Me(X.__webglFramebuffer[Xe],R,T,i.COLOR_ATTACHMENT0,be,Xe);else Me(X.__webglFramebuffer,R,T,i.COLOR_ATTACHMENT0,be,0);g(T)&&m(be),t.unbindTexture()}R.depthBuffer&&Ye(R)}function Q(R){const T=R.textures;for(let X=0,se=T.length;X<se;X++){const fe=T[X];if(g(fe)){const $=_(R),Be=n.get(fe).__webglTexture;t.bindTexture($,Be),m($),t.unbindTexture()}}}const w=[],ne=[];function oe(R){if(R.samples>0){if(ie(R)===!1){const T=R.textures,X=R.width,se=R.height;let fe=i.COLOR_BUFFER_BIT;const $=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Be=n.get(R),be=T.length>1;if(be)for(let H=0;H<T.length;H++)t.bindFramebuffer(i.FRAMEBUFFER,Be.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+H,i.RENDERBUFFER,null),t.bindFramebuffer(i.FRAMEBUFFER,Be.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+H,i.TEXTURE_2D,null,0);t.bindFramebuffer(i.READ_FRAMEBUFFER,Be.__webglMultisampledFramebuffer);const Xe=R.texture.mipmaps;Xe&&Xe.length>0?t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Be.__webglFramebuffer[0]):t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Be.__webglFramebuffer);for(let H=0;H<T.length;H++){if(R.resolveDepthBuffer&&(R.depthBuffer&&(fe|=i.DEPTH_BUFFER_BIT),R.stencilBuffer&&R.resolveStencilBuffer&&(fe|=i.STENCIL_BUFFER_BIT)),be){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,Be.__webglColorRenderbuffer[H]);const J=n.get(T[H]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,J,0)}i.blitFramebuffer(0,0,X,se,0,0,X,se,fe,i.NEAREST),l===!0&&(w.length=0,ne.length=0,w.push(i.COLOR_ATTACHMENT0+H),R.depthBuffer&&R.resolveDepthBuffer===!1&&(w.push($),ne.push($),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,ne)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,w))}if(t.bindFramebuffer(i.READ_FRAMEBUFFER,null),t.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),be)for(let H=0;H<T.length;H++){t.bindFramebuffer(i.FRAMEBUFFER,Be.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+H,i.RENDERBUFFER,Be.__webglColorRenderbuffer[H]);const J=n.get(T[H]).__webglTexture;t.bindFramebuffer(i.FRAMEBUFFER,Be.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+H,i.TEXTURE_2D,J,0)}t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Be.__webglMultisampledFramebuffer)}else if(R.depthBuffer&&R.resolveDepthBuffer===!1&&l){const T=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[T])}}}function he(R){return Math.min(s.maxSamples,R.samples)}function ie(R){const T=n.get(R);return R.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&T.__useRenderToTexture!==!1}function pe(R){const T=o.render.frame;u.get(R)!==T&&(u.set(R,T),R.update())}function te(R,T){const X=R.colorSpace,se=R.format,fe=R.type;return R.isCompressedTexture===!0||R.isVideoTexture===!0||X!==Xo&&X!==js&&(yt.getTransfer(X)===Ut?(se!==Yn||fe!==ls)&&ut("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):ln("WebGLTextures: Unsupported texture color space:",X)),T}function ve(R){return typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement?(c.width=R.naturalWidth||R.width,c.height=R.naturalHeight||R.height):typeof VideoFrame<"u"&&R instanceof VideoFrame?(c.width=R.displayWidth,c.height=R.displayHeight):(c.width=R.width,c.height=R.height),c}this.allocateTextureUnit=N,this.resetTextureUnits=B,this.setTexture2D=V,this.setTexture2DArray=F,this.setTexture3D=q,this.setTextureCube=G,this.rebindTextures=O,this.setupRenderTarget=z,this.updateRenderTargetMipmap=Q,this.updateMultisampleRenderTarget=oe,this.setupDepthRenderbuffer=Ye,this.setupFrameBufferTexture=Me,this.useMultisampledRTT=ie}function tx(i,e){function t(n,s=js){let r;const o=yt.getTransfer(s);if(n===ls)return i.UNSIGNED_BYTE;if(n===Kd)return i.UNSIGNED_SHORT_4_4_4_4;if(n===jd)return i.UNSIGNED_SHORT_5_5_5_1;if(n===Fg)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===Lg)return i.UNSIGNED_INT_10F_11F_11F_REV;if(n===Dg)return i.BYTE;if(n===Pg)return i.SHORT;if(n===Ya)return i.UNSIGNED_SHORT;if(n===Qd)return i.INT;if(n===Ii)return i.UNSIGNED_INT;if(n===ki)return i.FLOAT;if(n===Qr)return i.HALF_FLOAT;if(n===Bg)return i.ALPHA;if(n===Ug)return i.RGB;if(n===Yn)return i.RGBA;if(n===Wo)return i.DEPTH_COMPONENT;if(n===Ka)return i.DEPTH_STENCIL;if(n===Og)return i.RED;if(n===Wc)return i.RED_INTEGER;if(n===$d)return i.RG;if(n===Zd)return i.RG_INTEGER;if(n===Do)return i.RGBA_INTEGER;if(n===sc||n===rc||n===oc||n===ac)if(o===Ut)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(n===sc)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===rc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===oc)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===ac)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(n===sc)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===rc)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===oc)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===ac)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===Uf||n===Of||n===Nf||n===zf)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(n===Uf)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===Of)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===Nf)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===zf)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===kf||n===Hf||n===Vf)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(n===kf||n===Hf)return o===Ut?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(n===Vf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(n===Gf||n===Wf||n===Xf||n===qf||n===Yf||n===Qf||n===Kf||n===jf||n===$f||n===Zf||n===Jf||n===ed||n===td||n===nd)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(n===Gf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===Wf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===Xf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===qf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===Yf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===Qf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===Kf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===jf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===$f)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===Zf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===Jf)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===ed)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===td)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===nd)return o===Ut?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===id||n===sd||n===rd)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(n===id)return o===Ut?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===sd)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===rd)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===od||n===ad||n===ld||n===cd)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(n===od)return r.COMPRESSED_RED_RGTC1_EXT;if(n===ad)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===ld)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===cd)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===Qa?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:t}}const l1=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,c1=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;class u1{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const n=new Kg(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,n=new Kn({vertexShader:l1,fragmentShader:c1,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new un(new Yo(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class f1 extends Kr{constructor(e,t){super();const n=this;let s=null,r=1,o=null,a="local-floor",l=1,c=null,u=null,f=null,d=null,h=null,x=null;const p=typeof XRWebGLBinding<"u",g=new u1,m={},_=t.getContextAttributes();let v=null,A=null;const S=[],y=[],M=new Ke;let E=null;const b=new Ti;b.viewport=new Zt;const C=new Ti;C.viewport=new Zt;const P=[b,C],B=new Dy;let N=null,k=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(ee){let ue=S[ee];return ue===void 0&&(ue=new Ou,S[ee]=ue),ue.getTargetRaySpace()},this.getControllerGrip=function(ee){let ue=S[ee];return ue===void 0&&(ue=new Ou,S[ee]=ue),ue.getGripSpace()},this.getHand=function(ee){let ue=S[ee];return ue===void 0&&(ue=new Ou,S[ee]=ue),ue.getHandSpace()};function V(ee){const ue=y.indexOf(ee.inputSource);if(ue===-1)return;const Me=S[ue];Me!==void 0&&(Me.update(ee.inputSource,ee.frame,c||o),Me.dispatchEvent({type:ee.type,data:ee.inputSource}))}function F(){s.removeEventListener("select",V),s.removeEventListener("selectstart",V),s.removeEventListener("selectend",V),s.removeEventListener("squeeze",V),s.removeEventListener("squeezestart",V),s.removeEventListener("squeezeend",V),s.removeEventListener("end",F),s.removeEventListener("inputsourceschange",q);for(let ee=0;ee<S.length;ee++){const ue=y[ee];ue!==null&&(y[ee]=null,S[ee].disconnect(ue))}N=null,k=null,g.reset();for(const ee in m)delete m[ee];e.setRenderTarget(v),h=null,d=null,f=null,s=null,A=null,qe.stop(),n.isPresenting=!1,e.setPixelRatio(E),e.setSize(M.width,M.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(ee){r=ee,n.isPresenting===!0&&ut("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(ee){a=ee,n.isPresenting===!0&&ut("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||o},this.setReferenceSpace=function(ee){c=ee},this.getBaseLayer=function(){return d!==null?d:h},this.getBinding=function(){return f===null&&p&&(f=new XRWebGLBinding(s,t)),f},this.getFrame=function(){return x},this.getSession=function(){return s},this.setSession=async function(ee){if(s=ee,s!==null){if(v=e.getRenderTarget(),s.addEventListener("select",V),s.addEventListener("selectstart",V),s.addEventListener("selectend",V),s.addEventListener("squeeze",V),s.addEventListener("squeezestart",V),s.addEventListener("squeezeend",V),s.addEventListener("end",F),s.addEventListener("inputsourceschange",q),_.xrCompatible!==!0&&await t.makeXRCompatible(),E=e.getPixelRatio(),e.getSize(M),p&&"createProjectionLayer"in XRWebGLBinding.prototype){let Me=null,Ge=null,Ce=null;_.depth&&(Ce=_.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,Me=_.stencil?Ka:Wo,Ge=_.stencil?Qa:Ii);const Ye={colorFormat:t.RGBA8,depthFormat:Ce,scaleFactor:r};f=this.getBinding(),d=f.createProjectionLayer(Ye),s.updateRenderState({layers:[d]}),e.setPixelRatio(1),e.setSize(d.textureWidth,d.textureHeight,!1),A=new cr(d.textureWidth,d.textureHeight,{format:Yn,type:ls,depthTexture:new nh(d.textureWidth,d.textureHeight,Ge,void 0,void 0,void 0,void 0,void 0,void 0,Me),stencilBuffer:_.stencil,colorSpace:e.outputColorSpace,samples:_.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1})}else{const Me={antialias:_.antialias,alpha:!0,depth:_.depth,stencil:_.stencil,framebufferScaleFactor:r};h=new XRWebGLLayer(s,t,Me),s.updateRenderState({baseLayer:h}),e.setPixelRatio(1),e.setSize(h.framebufferWidth,h.framebufferHeight,!1),A=new cr(h.framebufferWidth,h.framebufferHeight,{format:Yn,type:ls,colorSpace:e.outputColorSpace,stencilBuffer:_.stencil,resolveDepthBuffer:h.ignoreDepthValues===!1,resolveStencilBuffer:h.ignoreDepthValues===!1})}A.isXRRenderTarget=!0,this.setFoveation(l),c=null,o=await s.requestReferenceSpace(a),qe.setContext(s),qe.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return g.getDepthTexture()};function q(ee){for(let ue=0;ue<ee.removed.length;ue++){const Me=ee.removed[ue],Ge=y.indexOf(Me);Ge>=0&&(y[Ge]=null,S[Ge].disconnect(Me))}for(let ue=0;ue<ee.added.length;ue++){const Me=ee.added[ue];let Ge=y.indexOf(Me);if(Ge===-1){for(let Ye=0;Ye<S.length;Ye++)if(Ye>=y.length){y.push(Me),Ge=Ye;break}else if(y[Ye]===null){y[Ye]=Me,Ge=Ye;break}if(Ge===-1)break}const Ce=S[Ge];Ce&&Ce.connect(Me)}}const G=new U,j=new U;function ce(ee,ue,Me){G.setFromMatrixPosition(ue.matrixWorld),j.setFromMatrixPosition(Me.matrixWorld);const Ge=G.distanceTo(j),Ce=ue.projectionMatrix.elements,Ye=Me.projectionMatrix.elements,O=Ce[14]/(Ce[10]-1),z=Ce[14]/(Ce[10]+1),Q=(Ce[9]+1)/Ce[5],w=(Ce[9]-1)/Ce[5],ne=(Ce[8]-1)/Ce[0],oe=(Ye[8]+1)/Ye[0],he=O*ne,ie=O*oe,pe=Ge/(-ne+oe),te=pe*-ne;if(ue.matrixWorld.decompose(ee.position,ee.quaternion,ee.scale),ee.translateX(te),ee.translateZ(pe),ee.matrixWorld.compose(ee.position,ee.quaternion,ee.scale),ee.matrixWorldInverse.copy(ee.matrixWorld).invert(),Ce[10]===-1)ee.projectionMatrix.copy(ue.projectionMatrix),ee.projectionMatrixInverse.copy(ue.projectionMatrixInverse);else{const ve=O+pe,R=z+pe,T=he-te,X=ie+(Ge-te),se=Q*z/R*ve,fe=w*z/R*ve;ee.projectionMatrix.makePerspective(T,X,se,fe,ve,R),ee.projectionMatrixInverse.copy(ee.projectionMatrix).invert()}}function ye(ee,ue){ue===null?ee.matrixWorld.copy(ee.matrix):ee.matrixWorld.multiplyMatrices(ue.matrixWorld,ee.matrix),ee.matrixWorldInverse.copy(ee.matrixWorld).invert()}this.updateCamera=function(ee){if(s===null)return;let ue=ee.near,Me=ee.far;g.texture!==null&&(g.depthNear>0&&(ue=g.depthNear),g.depthFar>0&&(Me=g.depthFar)),B.near=C.near=b.near=ue,B.far=C.far=b.far=Me,(N!==B.near||k!==B.far)&&(s.updateRenderState({depthNear:B.near,depthFar:B.far}),N=B.near,k=B.far),B.layers.mask=ee.layers.mask|6,b.layers.mask=B.layers.mask&3,C.layers.mask=B.layers.mask&5;const Ge=ee.parent,Ce=B.cameras;ye(B,Ge);for(let Ye=0;Ye<Ce.length;Ye++)ye(Ce[Ye],Ge);Ce.length===2?ce(B,b,C):B.projectionMatrix.copy(b.projectionMatrix),Ie(ee,B,Ge)};function Ie(ee,ue,Me){Me===null?ee.matrix.copy(ue.matrixWorld):(ee.matrix.copy(Me.matrixWorld),ee.matrix.invert(),ee.matrix.multiply(ue.matrixWorld)),ee.matrix.decompose(ee.position,ee.quaternion,ee.scale),ee.updateMatrixWorld(!0),ee.projectionMatrix.copy(ue.projectionMatrix),ee.projectionMatrixInverse.copy(ue.projectionMatrixInverse),ee.isPerspectiveCamera&&(ee.fov=$a*2*Math.atan(1/ee.projectionMatrix.elements[5]),ee.zoom=1)}this.getCamera=function(){return B},this.getFoveation=function(){if(!(d===null&&h===null))return l},this.setFoveation=function(ee){l=ee,d!==null&&(d.fixedFoveation=ee),h!==null&&h.fixedFoveation!==void 0&&(h.fixedFoveation=ee)},this.hasDepthSensing=function(){return g.texture!==null},this.getDepthSensingMesh=function(){return g.getMesh(B)},this.getCameraTexture=function(ee){return m[ee]};let Ne=null;function de(ee,ue){if(u=ue.getViewerPose(c||o),x=ue,u!==null){const Me=u.views;h!==null&&(e.setRenderTargetFramebuffer(A,h.framebuffer),e.setRenderTarget(A));let Ge=!1;Me.length!==B.cameras.length&&(B.cameras.length=0,Ge=!0);for(let z=0;z<Me.length;z++){const Q=Me[z];let w=null;if(h!==null)w=h.getViewport(Q);else{const oe=f.getViewSubImage(d,Q);w=oe.viewport,z===0&&(e.setRenderTargetTextures(A,oe.colorTexture,oe.depthStencilTexture),e.setRenderTarget(A))}let ne=P[z];ne===void 0&&(ne=new Ti,ne.layers.enable(z),ne.viewport=new Zt,P[z]=ne),ne.matrix.fromArray(Q.transform.matrix),ne.matrix.decompose(ne.position,ne.quaternion,ne.scale),ne.projectionMatrix.fromArray(Q.projectionMatrix),ne.projectionMatrixInverse.copy(ne.projectionMatrix).invert(),ne.viewport.set(w.x,w.y,w.width,w.height),z===0&&(B.matrix.copy(ne.matrix),B.matrix.decompose(B.position,B.quaternion,B.scale)),Ge===!0&&B.cameras.push(ne)}const Ce=s.enabledFeatures;if(Ce&&Ce.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&p){f=n.getBinding();const z=f.getDepthInformation(Me[0]);z&&z.isValid&&z.texture&&g.init(z,s.renderState)}if(Ce&&Ce.includes("camera-access")&&p){e.state.unbindTexture(),f=n.getBinding();for(let z=0;z<Me.length;z++){const Q=Me[z].camera;if(Q){let w=m[Q];w||(w=new Kg,m[Q]=w);const ne=f.getCameraImage(Q);w.sourceTexture=ne}}}}for(let Me=0;Me<S.length;Me++){const Ge=y[Me],Ce=S[Me];Ge!==null&&Ce!==void 0&&Ce.update(Ge,ue,c||o)}Ne&&Ne(ee,ue),ue.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:ue}),x=null}const qe=new jg;qe.setAnimationLoop(de),this.setAnimationLoop=function(ee){Ne=ee},this.dispose=function(){}}}const Cr=new Gi,d1=new rt;function h1(i,e){function t(g,m){g.matrixAutoUpdate===!0&&g.updateMatrix(),m.value.copy(g.matrix)}function n(g,m){m.color.getRGB(g.fogColor.value,Xg(i)),m.isFog?(g.fogNear.value=m.near,g.fogFar.value=m.far):m.isFogExp2&&(g.fogDensity.value=m.density)}function s(g,m,_,v,A){m.isMeshBasicMaterial||m.isMeshLambertMaterial?r(g,m):m.isMeshToonMaterial?(r(g,m),f(g,m)):m.isMeshPhongMaterial?(r(g,m),u(g,m)):m.isMeshStandardMaterial?(r(g,m),d(g,m),m.isMeshPhysicalMaterial&&h(g,m,A)):m.isMeshMatcapMaterial?(r(g,m),x(g,m)):m.isMeshDepthMaterial?r(g,m):m.isMeshDistanceMaterial?(r(g,m),p(g,m)):m.isMeshNormalMaterial?r(g,m):m.isLineBasicMaterial?(o(g,m),m.isLineDashedMaterial&&a(g,m)):m.isPointsMaterial?l(g,m,_,v):m.isSpriteMaterial?c(g,m):m.isShadowMaterial?(g.color.value.copy(m.color),g.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function r(g,m){g.opacity.value=m.opacity,m.color&&g.diffuse.value.copy(m.color),m.emissive&&g.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.bumpMap&&(g.bumpMap.value=m.bumpMap,t(m.bumpMap,g.bumpMapTransform),g.bumpScale.value=m.bumpScale,m.side===ti&&(g.bumpScale.value*=-1)),m.normalMap&&(g.normalMap.value=m.normalMap,t(m.normalMap,g.normalMapTransform),g.normalScale.value.copy(m.normalScale),m.side===ti&&g.normalScale.value.negate()),m.displacementMap&&(g.displacementMap.value=m.displacementMap,t(m.displacementMap,g.displacementMapTransform),g.displacementScale.value=m.displacementScale,g.displacementBias.value=m.displacementBias),m.emissiveMap&&(g.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,g.emissiveMapTransform)),m.specularMap&&(g.specularMap.value=m.specularMap,t(m.specularMap,g.specularMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest);const _=e.get(m),v=_.envMap,A=_.envMapRotation;v&&(g.envMap.value=v,Cr.copy(A),Cr.x*=-1,Cr.y*=-1,Cr.z*=-1,v.isCubeTexture&&v.isRenderTargetTexture===!1&&(Cr.y*=-1,Cr.z*=-1),g.envMapRotation.value.setFromMatrix4(d1.makeRotationFromEuler(Cr)),g.flipEnvMap.value=v.isCubeTexture&&v.isRenderTargetTexture===!1?-1:1,g.reflectivity.value=m.reflectivity,g.ior.value=m.ior,g.refractionRatio.value=m.refractionRatio),m.lightMap&&(g.lightMap.value=m.lightMap,g.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,g.lightMapTransform)),m.aoMap&&(g.aoMap.value=m.aoMap,g.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,g.aoMapTransform))}function o(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform))}function a(g,m){g.dashSize.value=m.dashSize,g.totalSize.value=m.dashSize+m.gapSize,g.scale.value=m.scale}function l(g,m,_,v){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.size.value=m.size*_,g.scale.value=v*.5,m.map&&(g.map.value=m.map,t(m.map,g.uvTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function c(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.rotation.value=m.rotation,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function u(g,m){g.specular.value.copy(m.specular),g.shininess.value=Math.max(m.shininess,1e-4)}function f(g,m){m.gradientMap&&(g.gradientMap.value=m.gradientMap)}function d(g,m){g.metalness.value=m.metalness,m.metalnessMap&&(g.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,g.metalnessMapTransform)),g.roughness.value=m.roughness,m.roughnessMap&&(g.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,g.roughnessMapTransform)),m.envMap&&(g.envMapIntensity.value=m.envMapIntensity)}function h(g,m,_){g.ior.value=m.ior,m.sheen>0&&(g.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),g.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(g.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,g.sheenColorMapTransform)),m.sheenRoughnessMap&&(g.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,g.sheenRoughnessMapTransform))),m.clearcoat>0&&(g.clearcoat.value=m.clearcoat,g.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(g.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,g.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(g.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,g.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(g.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,g.clearcoatNormalMapTransform),g.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===ti&&g.clearcoatNormalScale.value.negate())),m.dispersion>0&&(g.dispersion.value=m.dispersion),m.iridescence>0&&(g.iridescence.value=m.iridescence,g.iridescenceIOR.value=m.iridescenceIOR,g.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],g.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(g.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,g.iridescenceMapTransform)),m.iridescenceThicknessMap&&(g.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,g.iridescenceThicknessMapTransform))),m.transmission>0&&(g.transmission.value=m.transmission,g.transmissionSamplerMap.value=_.texture,g.transmissionSamplerSize.value.set(_.width,_.height),m.transmissionMap&&(g.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,g.transmissionMapTransform)),g.thickness.value=m.thickness,m.thicknessMap&&(g.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,g.thicknessMapTransform)),g.attenuationDistance.value=m.attenuationDistance,g.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(g.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(g.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,g.anisotropyMapTransform))),g.specularIntensity.value=m.specularIntensity,g.specularColor.value.copy(m.specularColor),m.specularColorMap&&(g.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,g.specularColorMapTransform)),m.specularIntensityMap&&(g.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,g.specularIntensityMapTransform))}function x(g,m){m.matcap&&(g.matcap.value=m.matcap)}function p(g,m){const _=e.get(m).light;g.referencePosition.value.setFromMatrixPosition(_.matrixWorld),g.nearDistance.value=_.shadow.camera.near,g.farDistance.value=_.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:s}}function p1(i,e,t,n){let s={},r={},o=[];const a=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function l(_,v){const A=v.program;n.uniformBlockBinding(_,A)}function c(_,v){let A=s[_.id];A===void 0&&(x(_),A=u(_),s[_.id]=A,_.addEventListener("dispose",g));const S=v.program;n.updateUBOMapping(_,S);const y=e.render.frame;r[_.id]!==y&&(d(_),r[_.id]=y)}function u(_){const v=f();_.__bindingPointIndex=v;const A=i.createBuffer(),S=_.__size,y=_.usage;return i.bindBuffer(i.UNIFORM_BUFFER,A),i.bufferData(i.UNIFORM_BUFFER,S,y),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,v,A),A}function f(){for(let _=0;_<a;_++)if(o.indexOf(_)===-1)return o.push(_),_;return ln("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function d(_){const v=s[_.id],A=_.uniforms,S=_.__cache;i.bindBuffer(i.UNIFORM_BUFFER,v);for(let y=0,M=A.length;y<M;y++){const E=Array.isArray(A[y])?A[y]:[A[y]];for(let b=0,C=E.length;b<C;b++){const P=E[b];if(h(P,y,b,S)===!0){const B=P.__offset,N=Array.isArray(P.value)?P.value:[P.value];let k=0;for(let V=0;V<N.length;V++){const F=N[V],q=p(F);typeof F=="number"||typeof F=="boolean"?(P.__data[0]=F,i.bufferSubData(i.UNIFORM_BUFFER,B+k,P.__data)):F.isMatrix3?(P.__data[0]=F.elements[0],P.__data[1]=F.elements[1],P.__data[2]=F.elements[2],P.__data[3]=0,P.__data[4]=F.elements[3],P.__data[5]=F.elements[4],P.__data[6]=F.elements[5],P.__data[7]=0,P.__data[8]=F.elements[6],P.__data[9]=F.elements[7],P.__data[10]=F.elements[8],P.__data[11]=0):(F.toArray(P.__data,k),k+=q.storage/Float32Array.BYTES_PER_ELEMENT)}i.bufferSubData(i.UNIFORM_BUFFER,B,P.__data)}}}i.bindBuffer(i.UNIFORM_BUFFER,null)}function h(_,v,A,S){const y=_.value,M=v+"_"+A;if(S[M]===void 0)return typeof y=="number"||typeof y=="boolean"?S[M]=y:S[M]=y.clone(),!0;{const E=S[M];if(typeof y=="number"||typeof y=="boolean"){if(E!==y)return S[M]=y,!0}else if(E.equals(y)===!1)return E.copy(y),!0}return!1}function x(_){const v=_.uniforms;let A=0;const S=16;for(let M=0,E=v.length;M<E;M++){const b=Array.isArray(v[M])?v[M]:[v[M]];for(let C=0,P=b.length;C<P;C++){const B=b[C],N=Array.isArray(B.value)?B.value:[B.value];for(let k=0,V=N.length;k<V;k++){const F=N[k],q=p(F),G=A%S,j=G%q.boundary,ce=G+j;A+=j,ce!==0&&S-ce<q.storage&&(A+=S-ce),B.__data=new Float32Array(q.storage/Float32Array.BYTES_PER_ELEMENT),B.__offset=A,A+=q.storage}}}const y=A%S;return y>0&&(A+=S-y),_.__size=A,_.__cache={},this}function p(_){const v={boundary:0,storage:0};return typeof _=="number"||typeof _=="boolean"?(v.boundary=4,v.storage=4):_.isVector2?(v.boundary=8,v.storage=8):_.isVector3||_.isColor?(v.boundary=16,v.storage=12):_.isVector4?(v.boundary=16,v.storage=16):_.isMatrix3?(v.boundary=48,v.storage=48):_.isMatrix4?(v.boundary=64,v.storage=64):_.isTexture?ut("WebGLRenderer: Texture samplers can not be part of an uniforms group."):ut("WebGLRenderer: Unsupported uniform value type.",_),v}function g(_){const v=_.target;v.removeEventListener("dispose",g);const A=o.indexOf(v.__bindingPointIndex);o.splice(A,1),i.deleteBuffer(s[v.id]),delete s[v.id],delete r[v.id]}function m(){for(const _ in s)i.deleteBuffer(s[_]);o=[],s={},r={}}return{bind:l,update:c,dispose:m}}const m1=new Uint16Array([11481,15204,11534,15171,11808,15015,12385,14843,12894,14716,13396,14600,13693,14483,13976,14366,14237,14171,14405,13961,14511,13770,14605,13598,14687,13444,14760,13305,14822,13066,14876,12857,14923,12675,14963,12517,14997,12379,15025,12230,15049,12023,15070,11843,15086,11687,15100,11551,15111,11433,15120,11330,15127,11217,15132,11060,15135,10922,15138,10801,15139,10695,15139,10600,13012,14923,13020,14917,13064,14886,13176,14800,13349,14666,13513,14526,13724,14398,13960,14230,14200,14020,14383,13827,14488,13651,14583,13491,14667,13348,14740,13132,14803,12908,14856,12713,14901,12542,14938,12394,14968,12241,14992,12017,15010,11822,15024,11654,15034,11507,15041,11380,15044,11269,15044,11081,15042,10913,15037,10764,15031,10635,15023,10520,15014,10419,15003,10330,13657,14676,13658,14673,13670,14660,13698,14622,13750,14547,13834,14442,13956,14317,14112,14093,14291,13889,14407,13704,14499,13538,14586,13389,14664,13201,14733,12966,14792,12758,14842,12577,14882,12418,14915,12272,14940,12033,14959,11826,14972,11646,14980,11490,14983,11355,14983,11212,14979,11008,14971,10830,14961,10675,14950,10540,14936,10420,14923,10315,14909,10204,14894,10041,14089,14460,14090,14459,14096,14452,14112,14431,14141,14388,14186,14305,14252,14130,14341,13941,14399,13756,14467,13585,14539,13430,14610,13272,14677,13026,14737,12808,14790,12617,14833,12449,14869,12303,14896,12065,14916,11845,14929,11655,14937,11490,14939,11347,14936,11184,14930,10970,14921,10783,14912,10621,14900,10480,14885,10356,14867,10247,14848,10062,14827,9894,14805,9745,14400,14208,14400,14206,14402,14198,14406,14174,14415,14122,14427,14035,14444,13913,14469,13767,14504,13613,14548,13463,14598,13324,14651,13082,14704,12858,14752,12658,14795,12483,14831,12330,14860,12106,14881,11875,14895,11675,14903,11501,14905,11351,14903,11178,14900,10953,14892,10757,14880,10589,14865,10442,14847,10313,14827,10162,14805,9965,14782,9792,14757,9642,14731,9507,14562,13883,14562,13883,14563,13877,14566,13862,14570,13830,14576,13773,14584,13689,14595,13582,14613,13461,14637,13336,14668,13120,14704,12897,14741,12695,14776,12516,14808,12358,14835,12150,14856,11910,14870,11701,14878,11519,14882,11361,14884,11187,14880,10951,14871,10748,14858,10572,14842,10418,14823,10286,14801,10099,14777,9897,14751,9722,14725,9567,14696,9430,14666,9309,14702,13604,14702,13604,14702,13600,14703,13591,14705,13570,14707,13533,14709,13477,14712,13400,14718,13305,14727,13106,14743,12907,14762,12716,14784,12539,14807,12380,14827,12190,14844,11943,14855,11727,14863,11539,14870,11376,14871,11204,14868,10960,14858,10748,14845,10565,14829,10406,14809,10269,14786,10058,14761,9852,14734,9671,14705,9512,14674,9374,14641,9253,14608,9076,14821,13366,14821,13365,14821,13364,14821,13358,14821,13344,14821,13320,14819,13252,14817,13145,14815,13011,14814,12858,14817,12698,14823,12539,14832,12389,14841,12214,14850,11968,14856,11750,14861,11558,14866,11390,14867,11226,14862,10972,14853,10754,14840,10565,14823,10401,14803,10259,14780,10032,14754,9820,14725,9635,14694,9473,14661,9333,14627,9203,14593,8988,14557,8798,14923,13014,14922,13014,14922,13012,14922,13004,14920,12987,14919,12957,14915,12907,14909,12834,14902,12738,14894,12623,14888,12498,14883,12370,14880,12203,14878,11970,14875,11759,14873,11569,14874,11401,14872,11243,14865,10986,14855,10762,14842,10568,14825,10401,14804,10255,14781,10017,14754,9799,14725,9611,14692,9445,14658,9301,14623,9139,14587,8920,14548,8729,14509,8562,15008,12672,15008,12672,15008,12671,15007,12667,15005,12656,15001,12637,14997,12605,14989,12556,14978,12490,14966,12407,14953,12313,14940,12136,14927,11934,14914,11742,14903,11563,14896,11401,14889,11247,14879,10992,14866,10767,14851,10570,14833,10400,14812,10252,14789,10007,14761,9784,14731,9592,14698,9424,14663,9279,14627,9088,14588,8868,14548,8676,14508,8508,14467,8360,15080,12386,15080,12386,15079,12385,15078,12383,15076,12378,15072,12367,15066,12347,15057,12315,15045,12253,15030,12138,15012,11998,14993,11845,14972,11685,14951,11530,14935,11383,14920,11228,14904,10981,14887,10762,14870,10567,14850,10397,14827,10248,14803,9997,14774,9771,14743,9578,14710,9407,14674,9259,14637,9048,14596,8826,14555,8632,14514,8464,14471,8317,14427,8182,15139,12008,15139,12008,15138,12008,15137,12007,15135,12003,15130,11990,15124,11969,15115,11929,15102,11872,15086,11794,15064,11693,15041,11581,15013,11459,14987,11336,14966,11170,14944,10944,14921,10738,14898,10552,14875,10387,14850,10239,14824,9983,14794,9758,14762,9563,14728,9392,14692,9244,14653,9014,14611,8791,14569,8597,14526,8427,14481,8281,14436,8110,14391,7885,15188,11617,15188,11617,15187,11617,15186,11618,15183,11617,15179,11612,15173,11601,15163,11581,15150,11546,15133,11495,15110,11427,15083,11346,15051,11246,15024,11057,14996,10868,14967,10687,14938,10517,14911,10362,14882,10206,14853,9956,14821,9737,14787,9543,14752,9375,14715,9228,14675,8980,14632,8760,14589,8565,14544,8395,14498,8248,14451,8049,14404,7824,14357,7630,15228,11298,15228,11298,15227,11299,15226,11301,15223,11303,15219,11302,15213,11299,15204,11290,15191,11271,15174,11217,15150,11129,15119,11015,15087,10886,15057,10744,15024,10599,14990,10455,14957,10318,14924,10143,14891,9911,14856,9701,14820,9516,14782,9352,14744,9200,14703,8946,14659,8725,14615,8533,14568,8366,14521,8220,14472,7992,14423,7770,14374,7578,14315,7408,15260,10819,15260,10819,15259,10822,15258,10826,15256,10832,15251,10836,15246,10841,15237,10838,15225,10821,15207,10788,15183,10734,15151,10660,15120,10571,15087,10469,15049,10359,15012,10249,14974,10041,14937,9837,14900,9647,14860,9475,14820,9320,14779,9147,14736,8902,14691,8688,14646,8499,14598,8335,14549,8189,14499,7940,14448,7720,14397,7529,14347,7363,14256,7218,15285,10410,15285,10411,15285,10413,15284,10418,15282,10425,15278,10434,15272,10442,15264,10449,15252,10445,15235,10433,15210,10403,15179,10358,15149,10301,15113,10218,15073,10059,15033,9894,14991,9726,14951,9565,14909,9413,14865,9273,14822,9073,14777,8845,14730,8641,14682,8459,14633,8300,14583,8129,14531,7883,14479,7670,14426,7482,14373,7321,14305,7176,14201,6939,15305,9939,15305,9940,15305,9945,15304,9955,15302,9967,15298,9989,15293,10010,15286,10033,15274,10044,15258,10045,15233,10022,15205,9975,15174,9903,15136,9808,15095,9697,15053,9578,15009,9451,14965,9327,14918,9198,14871,8973,14825,8766,14775,8579,14725,8408,14675,8259,14622,8058,14569,7821,14515,7615,14460,7435,14405,7276,14350,7108,14256,6866,14149,6653,15321,9444,15321,9445,15321,9448,15320,9458,15317,9470,15314,9490,15310,9515,15302,9540,15292,9562,15276,9579,15251,9577,15226,9559,15195,9519,15156,9463,15116,9389,15071,9304,15025,9208,14978,9023,14927,8838,14878,8661,14827,8496,14774,8344,14722,8206,14667,7973,14612,7749,14556,7555,14499,7382,14443,7229,14385,7025,14322,6791,14210,6588,14100,6409,15333,8920,15333,8921,15332,8927,15332,8943,15329,8965,15326,9002,15322,9048,15316,9106,15307,9162,15291,9204,15267,9221,15244,9221,15212,9196,15175,9134,15133,9043,15088,8930,15040,8801,14990,8665,14938,8526,14886,8391,14830,8261,14775,8087,14719,7866,14661,7664,14603,7482,14544,7322,14485,7178,14426,6936,14367,6713,14281,6517,14166,6348,14054,6198,15341,8360,15341,8361,15341,8366,15341,8379,15339,8399,15336,8431,15332,8473,15326,8527,15318,8585,15302,8632,15281,8670,15258,8690,15227,8690,15191,8664,15149,8612,15104,8543,15055,8456,15001,8360,14948,8259,14892,8122,14834,7923,14776,7734,14716,7558,14656,7397,14595,7250,14534,7070,14472,6835,14410,6628,14350,6443,14243,6283,14125,6135,14010,5889,15348,7715,15348,7717,15348,7725,15347,7745,15345,7780,15343,7836,15339,7905,15334,8e3,15326,8103,15310,8193,15293,8239,15270,8270,15240,8287,15204,8283,15163,8260,15118,8223,15067,8143,15014,8014,14958,7873,14899,7723,14839,7573,14778,7430,14715,7293,14652,7164,14588,6931,14524,6720,14460,6531,14396,6362,14330,6210,14207,6015,14086,5781,13969,5576,15352,7114,15352,7116,15352,7128,15352,7159,15350,7195,15348,7237,15345,7299,15340,7374,15332,7457,15317,7544,15301,7633,15280,7703,15251,7754,15216,7775,15176,7767,15131,7733,15079,7670,15026,7588,14967,7492,14906,7387,14844,7278,14779,7171,14714,6965,14648,6770,14581,6587,14515,6420,14448,6269,14382,6123,14299,5881,14172,5665,14049,5477,13929,5310,15355,6329,15355,6330,15355,6339,15355,6362,15353,6410,15351,6472,15349,6572,15344,6688,15337,6835,15323,6985,15309,7142,15287,7220,15260,7277,15226,7310,15188,7326,15142,7318,15090,7285,15036,7239,14976,7177,14914,7045,14849,6892,14782,6736,14714,6581,14645,6433,14576,6293,14506,6164,14438,5946,14369,5733,14270,5540,14140,5369,14014,5216,13892,5043,15357,5483,15357,5484,15357,5496,15357,5528,15356,5597,15354,5692,15351,5835,15347,6011,15339,6195,15328,6317,15314,6446,15293,6566,15268,6668,15235,6746,15197,6796,15152,6811,15101,6790,15046,6748,14985,6673,14921,6583,14854,6479,14785,6371,14714,6259,14643,6149,14571,5946,14499,5750,14428,5567,14358,5401,14242,5250,14109,5111,13980,4870,13856,4657,15359,4555,15359,4557,15358,4573,15358,4633,15357,4715,15355,4841,15353,5061,15349,5216,15342,5391,15331,5577,15318,5770,15299,5967,15274,6150,15243,6223,15206,6280,15161,6310,15111,6317,15055,6300,14994,6262,14928,6208,14860,6141,14788,5994,14715,5838,14641,5684,14566,5529,14492,5384,14418,5247,14346,5121,14216,4892,14079,4682,13948,4496,13822,4330,15359,3498,15359,3501,15359,3520,15359,3598,15358,3719,15356,3860,15355,4137,15351,4305,15344,4563,15334,4809,15321,5116,15303,5273,15280,5418,15250,5547,15214,5653,15170,5722,15120,5761,15064,5763,15002,5733,14935,5673,14865,5597,14792,5504,14716,5400,14640,5294,14563,5185,14486,5041,14410,4841,14335,4655,14191,4482,14051,4325,13918,4183,13790,4012,15360,2282,15360,2285,15360,2306,15360,2401,15359,2547,15357,2748,15355,3103,15352,3349,15345,3675,15336,4020,15324,4272,15307,4496,15285,4716,15255,4908,15220,5086,15178,5170,15128,5214,15072,5234,15010,5231,14943,5206,14871,5166,14796,5102,14718,4971,14639,4833,14559,4687,14480,4541,14402,4401,14315,4268,14167,4142,14025,3958,13888,3747,13759,3556,15360,923,15360,925,15360,946,15360,1052,15359,1214,15357,1494,15356,1892,15352,2274,15346,2663,15338,3099,15326,3393,15309,3679,15288,3980,15260,4183,15226,4325,15185,4437,15136,4517,15080,4570,15018,4591,14950,4581,14877,4545,14800,4485,14720,4411,14638,4325,14556,4231,14475,4136,14395,3988,14297,3803,14145,3628,13999,3465,13861,3314,13729,3177,15360,263,15360,264,15360,272,15360,325,15359,407,15358,548,15356,780,15352,1144,15347,1580,15339,2099,15328,2425,15312,2795,15292,3133,15264,3329,15232,3517,15191,3689,15143,3819,15088,3923,15025,3978,14956,3999,14882,3979,14804,3931,14722,3855,14639,3756,14554,3645,14470,3529,14388,3409,14279,3289,14124,3173,13975,3055,13834,2848,13701,2658,15360,49,15360,49,15360,52,15360,75,15359,111,15358,201,15356,283,15353,519,15348,726,15340,1045,15329,1415,15314,1795,15295,2173,15269,2410,15237,2649,15197,2866,15150,3054,15095,3140,15032,3196,14963,3228,14888,3236,14808,3224,14725,3191,14639,3146,14553,3088,14466,2976,14382,2836,14262,2692,14103,2549,13952,2409,13808,2278,13674,2154,15360,4,15360,4,15360,4,15360,13,15359,33,15358,59,15357,112,15353,199,15348,302,15341,456,15331,628,15316,827,15297,1082,15272,1332,15241,1601,15202,1851,15156,2069,15101,2172,15039,2256,14970,2314,14894,2348,14813,2358,14728,2344,14640,2311,14551,2263,14463,2203,14376,2133,14247,2059,14084,1915,13930,1761,13784,1609,13648,1464,15360,0,15360,0,15360,0,15360,3,15359,18,15358,26,15357,53,15354,80,15348,97,15341,165,15332,238,15318,326,15299,427,15275,529,15245,654,15207,771,15161,885,15108,994,15046,1089,14976,1170,14900,1229,14817,1266,14731,1284,14641,1282,14550,1260,14460,1223,14370,1174,14232,1116,14066,1050,13909,981,13761,910,13623,839]);let _s=null;function g1(){return _s===null&&(_s=new Ss(m1,32,32,$d,Qr),_s.minFilter=Ri,_s.magFilter=Ri,_s.wrapS=Is,_s.wrapT=Is,_s.generateMipmaps=!1,_s.needsUpdate=!0),_s}class x1{constructor(e={}){const{canvas:t=RS(),context:n=null,depth:s=!0,stencil:r=!1,alpha:o=!1,antialias:a=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:u="default",failIfMajorPerformanceCaveat:f=!1,reversedDepthBuffer:d=!1}=e;this.isWebGLRenderer=!0;let h;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");h=n.getContextAttributes().alpha}else h=o;const x=new Set([Do,Zd,Wc]),p=new Set([ls,Ii,Ya,Qa,Kd,jd]),g=new Uint32Array(4),m=new Int32Array(4);let _=null,v=null;const A=[],S=[];this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=sr,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const y=this;let M=!1;this._outputColorSpace=Ci;let E=0,b=0,C=null,P=-1,B=null;const N=new Zt,k=new Zt;let V=null;const F=new vt(0);let q=0,G=t.width,j=t.height,ce=1,ye=null,Ie=null;const Ne=new Zt(0,0,G,j),de=new Zt(0,0,G,j);let qe=!1;const ee=new Qg;let ue=!1,Me=!1;const Ge=new rt,Ce=new U,Ye=new Zt,O={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let z=!1;function Q(){return C===null?ce:1}let w=n;function ne(D,Y){return t.getContext(D,Y)}try{const D={alpha:!0,depth:s,stencil:r,antialias:a,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:u,failIfMajorPerformanceCaveat:f};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${Yd}`),t.addEventListener("webglcontextlost",me,!1),t.addEventListener("webglcontextrestored",ge,!1),t.addEventListener("webglcontextcreationerror",We,!1),w===null){const Y="webgl2";if(w=ne(Y,D),w===null)throw ne(Y)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(D){throw D("WebGLRenderer: "+D.message),D}let oe,he,ie,pe,te,ve,R,T,X,se,fe,$,Be,be,Xe,H,J,_e,we,Le,Re,$e,W,Fe;function Ae(){oe=new TC(w),oe.init(),$e=new tx(w,oe),he=new xC(w,oe,e,$e),ie=new o1(w,oe),he.reversedDepthBuffer&&d&&ie.buffers.depth.setReversed(!0),pe=new RC(w),te=new YT,ve=new a1(w,oe,ie,te,he,$e,pe),R=new vC(y),T=new CC(y),X=new Fy(w),W=new mC(w,X),se=new EC(w,X,pe,W),fe=new DC(w,se,X,pe),we=new IC(w,he,ve),H=new _C(te),$=new qT(y,R,T,oe,he,W,H),Be=new h1(y,te),be=new KT,Xe=new t1(oe),_e=new pC(y,R,T,ie,fe,h,l),J=new s1(y,fe,he),Fe=new p1(w,pe,he,ie),Le=new gC(w,oe,pe),Re=new wC(w,oe,pe),pe.programs=$.programs,y.capabilities=he,y.extensions=oe,y.properties=te,y.renderLists=be,y.shadowMap=J,y.state=ie,y.info=pe}Ae();const xe=new f1(y,w);this.xr=xe,this.getContext=function(){return w},this.getContextAttributes=function(){return w.getContextAttributes()},this.forceContextLoss=function(){const D=oe.get("WEBGL_lose_context");D&&D.loseContext()},this.forceContextRestore=function(){const D=oe.get("WEBGL_lose_context");D&&D.restoreContext()},this.getPixelRatio=function(){return ce},this.setPixelRatio=function(D){D!==void 0&&(ce=D,this.setSize(G,j,!1))},this.getSize=function(D){return D.set(G,j)},this.setSize=function(D,Y,le=!0){if(xe.isPresenting){ut("WebGLRenderer: Can't change size while VR device is presenting.");return}G=D,j=Y,t.width=Math.floor(D*ce),t.height=Math.floor(Y*ce),le===!0&&(t.style.width=D+"px",t.style.height=Y+"px"),this.setViewport(0,0,D,Y)},this.getDrawingBufferSize=function(D){return D.set(G*ce,j*ce).floor()},this.setDrawingBufferSize=function(D,Y,le){G=D,j=Y,ce=le,t.width=Math.floor(D*le),t.height=Math.floor(Y*le),this.setViewport(0,0,D,Y)},this.getCurrentViewport=function(D){return D.copy(N)},this.getViewport=function(D){return D.copy(Ne)},this.setViewport=function(D,Y,le,ae){D.isVector4?Ne.set(D.x,D.y,D.z,D.w):Ne.set(D,Y,le,ae),ie.viewport(N.copy(Ne).multiplyScalar(ce).round())},this.getScissor=function(D){return D.copy(de)},this.setScissor=function(D,Y,le,ae){D.isVector4?de.set(D.x,D.y,D.z,D.w):de.set(D,Y,le,ae),ie.scissor(k.copy(de).multiplyScalar(ce).round())},this.getScissorTest=function(){return qe},this.setScissorTest=function(D){ie.setScissorTest(qe=D)},this.setOpaqueSort=function(D){ye=D},this.setTransparentSort=function(D){Ie=D},this.getClearColor=function(D){return D.copy(_e.getClearColor())},this.setClearColor=function(){_e.setClearColor(...arguments)},this.getClearAlpha=function(){return _e.getClearAlpha()},this.setClearAlpha=function(){_e.setClearAlpha(...arguments)},this.clear=function(D=!0,Y=!0,le=!0){let ae=0;if(D){let K=!1;if(C!==null){const Ee=C.texture.format;K=x.has(Ee)}if(K){const Ee=C.texture.type,Ue=p.has(Ee),je=_e.getClearColor(),Ve=_e.getClearAlpha(),st=je.r,ot=je.g,Je=je.b;Ue?(g[0]=st,g[1]=ot,g[2]=Je,g[3]=Ve,w.clearBufferuiv(w.COLOR,0,g)):(m[0]=st,m[1]=ot,m[2]=Je,m[3]=Ve,w.clearBufferiv(w.COLOR,0,m))}else ae|=w.COLOR_BUFFER_BIT}Y&&(ae|=w.DEPTH_BUFFER_BIT),le&&(ae|=w.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),w.clear(ae)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){t.removeEventListener("webglcontextlost",me,!1),t.removeEventListener("webglcontextrestored",ge,!1),t.removeEventListener("webglcontextcreationerror",We,!1),_e.dispose(),be.dispose(),Xe.dispose(),te.dispose(),R.dispose(),T.dispose(),fe.dispose(),W.dispose(),Fe.dispose(),$.dispose(),xe.dispose(),xe.removeEventListener("sessionstart",us),xe.removeEventListener("sessionend",sa),Wi.stop()};function me(D){D.preventDefault(),Dp("WebGLRenderer: Context Lost."),M=!0}function ge(){Dp("WebGLRenderer: Context Restored."),M=!1;const D=pe.autoReset,Y=J.enabled,le=J.autoUpdate,ae=J.needsUpdate,K=J.type;Ae(),pe.autoReset=D,J.enabled=Y,J.autoUpdate=le,J.needsUpdate=ae,J.type=K}function We(D){ln("WebGLRenderer: A WebGL context could not be created. Reason: ",D.statusMessage)}function et(D){const Y=D.target;Y.removeEventListener("dispose",et),Dt(Y)}function Dt(D){Ct(D),te.remove(D)}function Ct(D){const Y=te.get(D).programs;Y!==void 0&&(Y.forEach(function(le){$.releaseProgram(le)}),D.isShaderMaterial&&$.releaseShaderCache(D))}this.renderBufferDirect=function(D,Y,le,ae,K,Ee){Y===null&&(Y=O);const Ue=K.isMesh&&K.matrixWorld.determinant()<0,je=Jc(D,Y,le,ae,K);ie.setMaterial(ae,Ue);let Ve=le.index,st=1;if(ae.wireframe===!0){if(Ve=se.getWireframeAttribute(le),Ve===void 0)return;st=2}const ot=le.drawRange,Je=le.attributes.position;let mt=ot.start*st,Tt=(ot.start+ot.count)*st;Ee!==null&&(mt=Math.max(mt,Ee.start*st),Tt=Math.min(Tt,(Ee.start+Ee.count)*st)),Ve!==null?(mt=Math.max(mt,0),Tt=Math.min(Tt,Ve.count)):Je!=null&&(mt=Math.max(mt,0),Tt=Math.min(Tt,Je.count));const Yt=Tt-mt;if(Yt<0||Yt===1/0)return;W.setup(K,ae,je,le,Ve);let Qt,Et=Le;if(Ve!==null&&(Qt=X.get(Ve),Et=Re,Et.setIndex(Qt)),K.isMesh)ae.wireframe===!0?(ie.setLineWidth(ae.wireframeLinewidth*Q()),Et.setMode(w.LINES)):Et.setMode(w.TRIANGLES);else if(K.isLine){let tt=ae.linewidth;tt===void 0&&(tt=1),ie.setLineWidth(tt*Q()),K.isLineSegments?Et.setMode(w.LINES):K.isLineLoop?Et.setMode(w.LINE_LOOP):Et.setMode(w.LINE_STRIP)}else K.isPoints?Et.setMode(w.POINTS):K.isSprite&&Et.setMode(w.TRIANGLES);if(K.isBatchedMesh)if(K._multiDrawInstances!==null)ja("WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),Et.renderMultiDrawInstances(K._multiDrawStarts,K._multiDrawCounts,K._multiDrawCount,K._multiDrawInstances);else if(oe.get("WEBGL_multi_draw"))Et.renderMultiDraw(K._multiDrawStarts,K._multiDrawCounts,K._multiDrawCount);else{const tt=K._multiDrawStarts,Xt=K._multiDrawCounts,it=K._multiDrawCount,dt=Ve?X.get(Ve).bytesPerElement:1,xt=te.get(ae).currentProgram.getUniforms();for(let Pt=0;Pt<it;Pt++)xt.setValue(w,"_gl_DrawID",Pt),Et.render(tt[Pt]/dt,Xt[Pt])}else if(K.isInstancedMesh)Et.renderInstances(mt,Yt,K.count);else if(le.isInstancedBufferGeometry){const tt=le._maxInstanceCount!==void 0?le._maxInstanceCount:1/0,Xt=Math.min(le.instanceCount,tt);Et.renderInstances(mt,Yt,Xt)}else Et.render(mt,Yt)};function Si(D,Y,le){D.transparent===!0&&D.side===Ei&&D.forceSinglePass===!1?(D.side=ti,D.needsUpdate=!0,zs(D,Y,le),D.side=as,D.needsUpdate=!0,zs(D,Y,le),D.side=Ei):zs(D,Y,le)}this.compile=function(D,Y,le=null){le===null&&(le=D),v=Xe.get(le),v.init(Y),S.push(v),le.traverseVisible(function(K){K.isLight&&K.layers.test(Y.layers)&&(v.pushLight(K),K.castShadow&&v.pushShadow(K))}),D!==le&&D.traverseVisible(function(K){K.isLight&&K.layers.test(Y.layers)&&(v.pushLight(K),K.castShadow&&v.pushShadow(K))}),v.setupLights();const ae=new Set;return D.traverse(function(K){if(!(K.isMesh||K.isPoints||K.isLine||K.isSprite))return;const Ee=K.material;if(Ee)if(Array.isArray(Ee))for(let Ue=0;Ue<Ee.length;Ue++){const je=Ee[Ue];Si(je,le,K),ae.add(je)}else Si(Ee,le,K),ae.add(Ee)}),v=S.pop(),ae},this.compileAsync=function(D,Y,le=null){const ae=this.compile(D,Y,le);return new Promise(K=>{function Ee(){if(ae.forEach(function(Ue){te.get(Ue).currentProgram.isReady()&&ae.delete(Ue)}),ae.size===0){K(D);return}setTimeout(Ee,10)}oe.get("KHR_parallel_shader_compile")!==null?Ee():setTimeout(Ee,10)})};let Nn=null;function Zc(D){Nn&&Nn(D)}function us(){Wi.stop()}function sa(){Wi.start()}const Wi=new jg;Wi.setAnimationLoop(Zc),typeof self<"u"&&Wi.setContext(self),this.setAnimationLoop=function(D){Nn=D,xe.setAnimationLoop(D),D===null?Wi.stop():Wi.start()},xe.addEventListener("sessionstart",us),xe.addEventListener("sessionend",sa),this.render=function(D,Y){if(Y!==void 0&&Y.isCamera!==!0){ln("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(M===!0)return;if(D.matrixWorldAutoUpdate===!0&&D.updateMatrixWorld(),Y.parent===null&&Y.matrixWorldAutoUpdate===!0&&Y.updateMatrixWorld(),xe.enabled===!0&&xe.isPresenting===!0&&(xe.cameraAutoUpdate===!0&&xe.updateCamera(Y),Y=xe.getCamera()),D.isScene===!0&&D.onBeforeRender(y,D,Y,C),v=Xe.get(D,S.length),v.init(Y),S.push(v),Ge.multiplyMatrices(Y.projectionMatrix,Y.matrixWorldInverse),ee.setFromProjectionMatrix(Ge,Ji,Y.reversedDepth),Me=this.localClippingEnabled,ue=H.init(this.clippingPlanes,Me),_=be.get(D,A.length),_.init(),A.push(_),xe.enabled===!0&&xe.isPresenting===!0){const Ee=y.xr.getDepthSensingMesh();Ee!==null&&ra(Ee,Y,-1/0,y.sortObjects)}ra(D,Y,0,y.sortObjects),_.finish(),y.sortObjects===!0&&_.sort(ye,Ie),z=xe.enabled===!1||xe.isPresenting===!1||xe.hasDepthSensing()===!1,z&&_e.addToRenderList(_,D),this.info.render.frame++,ue===!0&&H.beginShadows();const le=v.state.shadowsArray;J.render(le,D,Y),ue===!0&&H.endShadows(),this.info.autoReset===!0&&this.info.reset();const ae=_.opaque,K=_.transmissive;if(v.setupLights(),Y.isArrayCamera){const Ee=Y.cameras;if(K.length>0)for(let Ue=0,je=Ee.length;Ue<je;Ue++){const Ve=Ee[Ue];oa(ae,K,D,Ve)}z&&_e.render(D);for(let Ue=0,je=Ee.length;Ue<je;Ue++){const Ve=Ee[Ue];$r(_,D,Ve,Ve.viewport)}}else K.length>0&&oa(ae,K,D,Y),z&&_e.render(D),$r(_,D,Y);C!==null&&b===0&&(ve.updateMultisampleRenderTarget(C),ve.updateRenderTargetMipmap(C)),D.isScene===!0&&D.onAfterRender(y,D,Y),W.resetDefaultState(),P=-1,B=null,S.pop(),S.length>0?(v=S[S.length-1],ue===!0&&H.setGlobalState(y.clippingPlanes,v.state.camera)):v=null,A.pop(),A.length>0?_=A[A.length-1]:_=null};function ra(D,Y,le,ae){if(D.visible===!1)return;if(D.layers.test(Y.layers)){if(D.isGroup)le=D.renderOrder;else if(D.isLOD)D.autoUpdate===!0&&D.update(Y);else if(D.isLight)v.pushLight(D),D.castShadow&&v.pushShadow(D);else if(D.isSprite){if(!D.frustumCulled||ee.intersectsSprite(D)){ae&&Ye.setFromMatrixPosition(D.matrixWorld).applyMatrix4(Ge);const Ue=fe.update(D),je=D.material;je.visible&&_.push(D,Ue,je,le,Ye.z,null)}}else if((D.isMesh||D.isLine||D.isPoints)&&(!D.frustumCulled||ee.intersectsObject(D))){const Ue=fe.update(D),je=D.material;if(ae&&(D.boundingSphere!==void 0?(D.boundingSphere===null&&D.computeBoundingSphere(),Ye.copy(D.boundingSphere.center)):(Ue.boundingSphere===null&&Ue.computeBoundingSphere(),Ye.copy(Ue.boundingSphere.center)),Ye.applyMatrix4(D.matrixWorld).applyMatrix4(Ge)),Array.isArray(je)){const Ve=Ue.groups;for(let st=0,ot=Ve.length;st<ot;st++){const Je=Ve[st],mt=je[Je.materialIndex];mt&&mt.visible&&_.push(D,Ue,mt,le,Ye.z,Je)}}else je.visible&&_.push(D,Ue,je,le,Ye.z,null)}}const Ee=D.children;for(let Ue=0,je=Ee.length;Ue<je;Ue++)ra(Ee[Ue],Y,le,ae)}function $r(D,Y,le,ae){const{opaque:K,transmissive:Ee,transparent:Ue}=D;v.setupLightsView(le),ue===!0&&H.setGlobalState(y.clippingPlanes,le),ae&&ie.viewport(N.copy(ae)),K.length>0&&mr(K,Y,le),Ee.length>0&&mr(Ee,Y,le),Ue.length>0&&mr(Ue,Y,le),ie.buffers.depth.setTest(!0),ie.buffers.depth.setMask(!0),ie.buffers.color.setMask(!0),ie.setPolygonOffset(!1)}function oa(D,Y,le,ae){if((le.isScene===!0?le.overrideMaterial:null)!==null)return;v.state.transmissionRenderTarget[ae.id]===void 0&&(v.state.transmissionRenderTarget[ae.id]=new cr(1,1,{generateMipmaps:!0,type:oe.has("EXT_color_buffer_half_float")||oe.has("EXT_color_buffer_float")?Qr:ls,minFilter:Nr,samples:4,stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:yt.workingColorSpace}));const Ee=v.state.transmissionRenderTarget[ae.id],Ue=ae.viewport||N;Ee.setSize(Ue.z*y.transmissionResolutionScale,Ue.w*y.transmissionResolutionScale);const je=y.getRenderTarget(),Ve=y.getActiveCubeFace(),st=y.getActiveMipmapLevel();y.setRenderTarget(Ee),y.getClearColor(F),q=y.getClearAlpha(),q<1&&y.setClearColor(16777215,.5),y.clear(),z&&_e.render(le);const ot=y.toneMapping;y.toneMapping=sr;const Je=ae.viewport;if(ae.viewport!==void 0&&(ae.viewport=void 0),v.setupLightsView(ae),ue===!0&&H.setGlobalState(y.clippingPlanes,ae),mr(D,le,ae),ve.updateMultisampleRenderTarget(Ee),ve.updateRenderTargetMipmap(Ee),oe.has("WEBGL_multisampled_render_to_texture")===!1){let mt=!1;for(let Tt=0,Yt=Y.length;Tt<Yt;Tt++){const Qt=Y[Tt],{object:Et,geometry:tt,material:Xt,group:it}=Qt;if(Xt.side===Ei&&Et.layers.test(ae.layers)){const dt=Xt.side;Xt.side=ti,Xt.needsUpdate=!0,Zr(Et,le,ae,tt,Xt,it),Xt.side=dt,Xt.needsUpdate=!0,mt=!0}}mt===!0&&(ve.updateMultisampleRenderTarget(Ee),ve.updateRenderTargetMipmap(Ee))}y.setRenderTarget(je,Ve,st),y.setClearColor(F,q),Je!==void 0&&(ae.viewport=Je),y.toneMapping=ot}function mr(D,Y,le){const ae=Y.isScene===!0?Y.overrideMaterial:null;for(let K=0,Ee=D.length;K<Ee;K++){const Ue=D[K],{object:je,geometry:Ve,group:st}=Ue;let ot=Ue.material;ot.allowOverride===!0&&ae!==null&&(ot=ae),je.layers.test(le.layers)&&Zr(je,Y,le,Ve,ot,st)}}function Zr(D,Y,le,ae,K,Ee){D.onBeforeRender(y,Y,le,ae,K,Ee),D.modelViewMatrix.multiplyMatrices(le.matrixWorldInverse,D.matrixWorld),D.normalMatrix.getNormalMatrix(D.modelViewMatrix),K.onBeforeRender(y,Y,le,ae,D,Ee),K.transparent===!0&&K.side===Ei&&K.forceSinglePass===!1?(K.side=ti,K.needsUpdate=!0,y.renderBufferDirect(le,Y,ae,K,D,Ee),K.side=as,K.needsUpdate=!0,y.renderBufferDirect(le,Y,ae,K,D,Ee),K.side=Ei):y.renderBufferDirect(le,Y,ae,K,D,Ee),D.onAfterRender(y,Y,le,ae,K,Ee)}function zs(D,Y,le){Y.isScene!==!0&&(Y=O);const ae=te.get(D),K=v.state.lights,Ee=v.state.shadowsArray,Ue=K.state.version,je=$.getParameters(D,K.state,Ee,Y,le),Ve=$.getProgramCacheKey(je);let st=ae.programs;ae.environment=D.isMeshStandardMaterial?Y.environment:null,ae.fog=Y.fog,ae.envMap=(D.isMeshStandardMaterial?T:R).get(D.envMap||ae.environment),ae.envMapRotation=ae.environment!==null&&D.envMap===null?Y.environmentRotation:D.envMapRotation,st===void 0&&(D.addEventListener("dispose",et),st=new Map,ae.programs=st);let ot=st.get(Ve);if(ot!==void 0){if(ae.currentProgram===ot&&ae.lightsStateVersion===Ue)return pl(D,je),ot}else je.uniforms=$.getUniforms(D),D.onBeforeCompile(je,y),ot=$.acquireProgram(je,Ve),st.set(Ve,ot),ae.uniforms=je.uniforms;const Je=ae.uniforms;return(!D.isShaderMaterial&&!D.isRawShaderMaterial||D.clipping===!0)&&(Je.clippingPlanes=H.uniform),pl(D,je),ae.needsLights=tu(D),ae.lightsStateVersion=Ue,ae.needsLights&&(Je.ambientLightColor.value=K.state.ambient,Je.lightProbe.value=K.state.probe,Je.directionalLights.value=K.state.directional,Je.directionalLightShadows.value=K.state.directionalShadow,Je.spotLights.value=K.state.spot,Je.spotLightShadows.value=K.state.spotShadow,Je.rectAreaLights.value=K.state.rectArea,Je.ltc_1.value=K.state.rectAreaLTC1,Je.ltc_2.value=K.state.rectAreaLTC2,Je.pointLights.value=K.state.point,Je.pointLightShadows.value=K.state.pointShadow,Je.hemisphereLights.value=K.state.hemi,Je.directionalShadowMap.value=K.state.directionalShadowMap,Je.directionalShadowMatrix.value=K.state.directionalShadowMatrix,Je.spotShadowMap.value=K.state.spotShadowMap,Je.spotLightMatrix.value=K.state.spotLightMatrix,Je.spotLightMap.value=K.state.spotLightMap,Je.pointShadowMap.value=K.state.pointShadowMap,Je.pointShadowMatrix.value=K.state.pointShadowMatrix),ae.currentProgram=ot,ae.uniformsList=null,ot}function aa(D){if(D.uniformsList===null){const Y=D.currentProgram.getUniforms();D.uniformsList=lc.seqWithValue(Y.seq,D.uniforms)}return D.uniformsList}function pl(D,Y){const le=te.get(D);le.outputColorSpace=Y.outputColorSpace,le.batching=Y.batching,le.batchingColor=Y.batchingColor,le.instancing=Y.instancing,le.instancingColor=Y.instancingColor,le.instancingMorph=Y.instancingMorph,le.skinning=Y.skinning,le.morphTargets=Y.morphTargets,le.morphNormals=Y.morphNormals,le.morphColors=Y.morphColors,le.morphTargetsCount=Y.morphTargetsCount,le.numClippingPlanes=Y.numClippingPlanes,le.numIntersection=Y.numClipIntersection,le.vertexAlphas=Y.vertexAlphas,le.vertexTangents=Y.vertexTangents,le.toneMapping=Y.toneMapping}function Jc(D,Y,le,ae,K){Y.isScene!==!0&&(Y=O),ve.resetTextureUnits();const Ee=Y.fog,Ue=ae.isMeshStandardMaterial?Y.environment:null,je=C===null?y.outputColorSpace:C.isXRRenderTarget===!0?C.texture.colorSpace:Xo,Ve=(ae.isMeshStandardMaterial?T:R).get(ae.envMap||Ue),st=ae.vertexColors===!0&&!!le.attributes.color&&le.attributes.color.itemSize===4,ot=!!le.attributes.tangent&&(!!ae.normalMap||ae.anisotropy>0),Je=!!le.morphAttributes.position,mt=!!le.morphAttributes.normal,Tt=!!le.morphAttributes.color;let Yt=sr;ae.toneMapped&&(C===null||C.isXRRenderTarget===!0)&&(Yt=y.toneMapping);const Qt=le.morphAttributes.position||le.morphAttributes.normal||le.morphAttributes.color,Et=Qt!==void 0?Qt.length:0,tt=te.get(ae),Xt=v.state.lights;if(ue===!0&&(Me===!0||D!==B)){const bn=D===B&&ae.id===P;H.setState(ae,D,bn)}let it=!1;ae.version===tt.__version?(tt.needsLights&&tt.lightsStateVersion!==Xt.state.version||tt.outputColorSpace!==je||K.isBatchedMesh&&tt.batching===!1||!K.isBatchedMesh&&tt.batching===!0||K.isBatchedMesh&&tt.batchingColor===!0&&K.colorTexture===null||K.isBatchedMesh&&tt.batchingColor===!1&&K.colorTexture!==null||K.isInstancedMesh&&tt.instancing===!1||!K.isInstancedMesh&&tt.instancing===!0||K.isSkinnedMesh&&tt.skinning===!1||!K.isSkinnedMesh&&tt.skinning===!0||K.isInstancedMesh&&tt.instancingColor===!0&&K.instanceColor===null||K.isInstancedMesh&&tt.instancingColor===!1&&K.instanceColor!==null||K.isInstancedMesh&&tt.instancingMorph===!0&&K.morphTexture===null||K.isInstancedMesh&&tt.instancingMorph===!1&&K.morphTexture!==null||tt.envMap!==Ve||ae.fog===!0&&tt.fog!==Ee||tt.numClippingPlanes!==void 0&&(tt.numClippingPlanes!==H.numPlanes||tt.numIntersection!==H.numIntersection)||tt.vertexAlphas!==st||tt.vertexTangents!==ot||tt.morphTargets!==Je||tt.morphNormals!==mt||tt.morphColors!==Tt||tt.toneMapping!==Yt||tt.morphTargetsCount!==Et)&&(it=!0):(it=!0,tt.__version=ae.version);let dt=tt.currentProgram;it===!0&&(dt=zs(ae,Y,K));let xt=!1,Pt=!1,yi=!1;const zt=dt.getUniforms(),xn=tt.uniforms;if(ie.useProgram(dt.program)&&(xt=!0,Pt=!0,yi=!0),ae.id!==P&&(P=ae.id,Pt=!0),xt||B!==D){ie.buffers.depth.getReversed()&&D.reversedDepth!==!0&&(D._reversedDepth=!0,D.updateProjectionMatrix()),zt.setValue(w,"projectionMatrix",D.projectionMatrix),zt.setValue(w,"viewMatrix",D.matrixWorldInverse);const Rn=zt.map.cameraPosition;Rn!==void 0&&Rn.setValue(w,Ce.setFromMatrixPosition(D.matrixWorld)),he.logarithmicDepthBuffer&&zt.setValue(w,"logDepthBufFC",2/(Math.log(D.far+1)/Math.LN2)),(ae.isMeshPhongMaterial||ae.isMeshToonMaterial||ae.isMeshLambertMaterial||ae.isMeshBasicMaterial||ae.isMeshStandardMaterial||ae.isShaderMaterial)&&zt.setValue(w,"isOrthographic",D.isOrthographicCamera===!0),B!==D&&(B=D,Pt=!0,yi=!0)}if(K.isSkinnedMesh){zt.setOptional(w,K,"bindMatrix"),zt.setOptional(w,K,"bindMatrixInverse");const bn=K.skeleton;bn&&(bn.boneTexture===null&&bn.computeBoneTexture(),zt.setValue(w,"boneTexture",bn.boneTexture,ve))}K.isBatchedMesh&&(zt.setOptional(w,K,"batchingTexture"),zt.setValue(w,"batchingTexture",K._matricesTexture,ve),zt.setOptional(w,K,"batchingIdTexture"),zt.setValue(w,"batchingIdTexture",K._indirectTexture,ve),zt.setOptional(w,K,"batchingColorTexture"),K._colorsTexture!==null&&zt.setValue(w,"batchingColorTexture",K._colorsTexture,ve));const yn=le.morphAttributes;if((yn.position!==void 0||yn.normal!==void 0||yn.color!==void 0)&&we.update(K,le,dt),(Pt||tt.receiveShadow!==K.receiveShadow)&&(tt.receiveShadow=K.receiveShadow,zt.setValue(w,"receiveShadow",K.receiveShadow)),ae.isMeshGouraudMaterial&&ae.envMap!==null&&(xn.envMap.value=Ve,xn.flipEnvMap.value=Ve.isCubeTexture&&Ve.isRenderTargetTexture===!1?-1:1),ae.isMeshStandardMaterial&&ae.envMap===null&&Y.environment!==null&&(xn.envMapIntensity.value=Y.environmentIntensity),xn.dfgLUT!==void 0&&(xn.dfgLUT.value=g1()),Pt&&(zt.setValue(w,"toneMappingExposure",y.toneMappingExposure),tt.needsLights&&eu(xn,yi),Ee&&ae.fog===!0&&Be.refreshFogUniforms(xn,Ee),Be.refreshMaterialUniforms(xn,ae,ce,j,v.state.transmissionRenderTarget[D.id]),lc.upload(w,aa(tt),xn,ve)),ae.isShaderMaterial&&ae.uniformsNeedUpdate===!0&&(lc.upload(w,aa(tt),xn,ve),ae.uniformsNeedUpdate=!1),ae.isSpriteMaterial&&zt.setValue(w,"center",K.center),zt.setValue(w,"modelViewMatrix",K.modelViewMatrix),zt.setValue(w,"normalMatrix",K.normalMatrix),zt.setValue(w,"modelMatrix",K.matrixWorld),ae.isShaderMaterial||ae.isRawShaderMaterial){const bn=ae.uniformsGroups;for(let Rn=0,la=bn.length;Rn<la;Rn++){const Xi=bn[Rn];Fe.update(Xi,dt),Fe.bind(Xi,dt)}}return dt}function eu(D,Y){D.ambientLightColor.needsUpdate=Y,D.lightProbe.needsUpdate=Y,D.directionalLights.needsUpdate=Y,D.directionalLightShadows.needsUpdate=Y,D.pointLights.needsUpdate=Y,D.pointLightShadows.needsUpdate=Y,D.spotLights.needsUpdate=Y,D.spotLightShadows.needsUpdate=Y,D.rectAreaLights.needsUpdate=Y,D.hemisphereLights.needsUpdate=Y}function tu(D){return D.isMeshLambertMaterial||D.isMeshToonMaterial||D.isMeshPhongMaterial||D.isMeshStandardMaterial||D.isShadowMaterial||D.isShaderMaterial&&D.lights===!0}this.getActiveCubeFace=function(){return E},this.getActiveMipmapLevel=function(){return b},this.getRenderTarget=function(){return C},this.setRenderTargetTextures=function(D,Y,le){const ae=te.get(D);ae.__autoAllocateDepthBuffer=D.resolveDepthBuffer===!1,ae.__autoAllocateDepthBuffer===!1&&(ae.__useRenderToTexture=!1),te.get(D.texture).__webglTexture=Y,te.get(D.depthTexture).__webglTexture=ae.__autoAllocateDepthBuffer?void 0:le,ae.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(D,Y){const le=te.get(D);le.__webglFramebuffer=Y,le.__useDefaultFramebuffer=Y===void 0};const ml=w.createFramebuffer();this.setRenderTarget=function(D,Y=0,le=0){C=D,E=Y,b=le;let ae=!0,K=null,Ee=!1,Ue=!1;if(D){const Ve=te.get(D);if(Ve.__useDefaultFramebuffer!==void 0)ie.bindFramebuffer(w.FRAMEBUFFER,null),ae=!1;else if(Ve.__webglFramebuffer===void 0)ve.setupRenderTarget(D);else if(Ve.__hasExternalTextures)ve.rebindTextures(D,te.get(D.texture).__webglTexture,te.get(D.depthTexture).__webglTexture);else if(D.depthBuffer){const Je=D.depthTexture;if(Ve.__boundDepthTexture!==Je){if(Je!==null&&te.has(Je)&&(D.width!==Je.image.width||D.height!==Je.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");ve.setupDepthRenderbuffer(D)}}const st=D.texture;(st.isData3DTexture||st.isDataArrayTexture||st.isCompressedArrayTexture)&&(Ue=!0);const ot=te.get(D).__webglFramebuffer;D.isWebGLCubeRenderTarget?(Array.isArray(ot[Y])?K=ot[Y][le]:K=ot[Y],Ee=!0):D.samples>0&&ve.useMultisampledRTT(D)===!1?K=te.get(D).__webglMultisampledFramebuffer:Array.isArray(ot)?K=ot[le]:K=ot,N.copy(D.viewport),k.copy(D.scissor),V=D.scissorTest}else N.copy(Ne).multiplyScalar(ce).floor(),k.copy(de).multiplyScalar(ce).floor(),V=qe;if(le!==0&&(K=ml),ie.bindFramebuffer(w.FRAMEBUFFER,K)&&ae&&ie.drawBuffers(D,K),ie.viewport(N),ie.scissor(k),ie.setScissorTest(V),Ee){const Ve=te.get(D.texture);w.framebufferTexture2D(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_CUBE_MAP_POSITIVE_X+Y,Ve.__webglTexture,le)}else if(Ue){const Ve=Y;for(let st=0;st<D.textures.length;st++){const ot=te.get(D.textures[st]);w.framebufferTextureLayer(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0+st,ot.__webglTexture,le,Ve)}}else if(D!==null&&le!==0){const Ve=te.get(D.texture);w.framebufferTexture2D(w.FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,Ve.__webglTexture,le)}P=-1},this.readRenderTargetPixels=function(D,Y,le,ae,K,Ee,Ue,je=0){if(!(D&&D.isWebGLRenderTarget)){ln("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Ve=te.get(D).__webglFramebuffer;if(D.isWebGLCubeRenderTarget&&Ue!==void 0&&(Ve=Ve[Ue]),Ve){ie.bindFramebuffer(w.FRAMEBUFFER,Ve);try{const st=D.textures[je],ot=st.format,Je=st.type;if(!he.textureFormatReadable(ot)){ln("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!he.textureTypeReadable(Je)){ln("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}Y>=0&&Y<=D.width-ae&&le>=0&&le<=D.height-K&&(D.textures.length>1&&w.readBuffer(w.COLOR_ATTACHMENT0+je),w.readPixels(Y,le,ae,K,$e.convert(ot),$e.convert(Je),Ee))}finally{const st=C!==null?te.get(C).__webglFramebuffer:null;ie.bindFramebuffer(w.FRAMEBUFFER,st)}}},this.readRenderTargetPixelsAsync=async function(D,Y,le,ae,K,Ee,Ue,je=0){if(!(D&&D.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let Ve=te.get(D).__webglFramebuffer;if(D.isWebGLCubeRenderTarget&&Ue!==void 0&&(Ve=Ve[Ue]),Ve)if(Y>=0&&Y<=D.width-ae&&le>=0&&le<=D.height-K){ie.bindFramebuffer(w.FRAMEBUFFER,Ve);const st=D.textures[je],ot=st.format,Je=st.type;if(!he.textureFormatReadable(ot))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!he.textureTypeReadable(Je))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const mt=w.createBuffer();w.bindBuffer(w.PIXEL_PACK_BUFFER,mt),w.bufferData(w.PIXEL_PACK_BUFFER,Ee.byteLength,w.STREAM_READ),D.textures.length>1&&w.readBuffer(w.COLOR_ATTACHMENT0+je),w.readPixels(Y,le,ae,K,$e.convert(ot),$e.convert(Je),0);const Tt=C!==null?te.get(C).__webglFramebuffer:null;ie.bindFramebuffer(w.FRAMEBUFFER,Tt);const Yt=w.fenceSync(w.SYNC_GPU_COMMANDS_COMPLETE,0);return w.flush(),await IS(w,Yt,4),w.bindBuffer(w.PIXEL_PACK_BUFFER,mt),w.getBufferSubData(w.PIXEL_PACK_BUFFER,0,Ee),w.deleteBuffer(mt),w.deleteSync(Yt),Ee}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(D,Y=null,le=0){const ae=Math.pow(2,-le),K=Math.floor(D.image.width*ae),Ee=Math.floor(D.image.height*ae),Ue=Y!==null?Y.x:0,je=Y!==null?Y.y:0;ve.setTexture2D(D,0),w.copyTexSubImage2D(w.TEXTURE_2D,le,0,0,Ue,je,K,Ee),ie.unbindTexture()};const nu=w.createFramebuffer(),gl=w.createFramebuffer();this.copyTextureToTexture=function(D,Y,le=null,ae=null,K=0,Ee=null){Ee===null&&(K!==0?(ja("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."),Ee=K,K=0):Ee=0);let Ue,je,Ve,st,ot,Je,mt,Tt,Yt;const Qt=D.isCompressedTexture?D.mipmaps[Ee]:D.image;if(le!==null)Ue=le.max.x-le.min.x,je=le.max.y-le.min.y,Ve=le.isBox3?le.max.z-le.min.z:1,st=le.min.x,ot=le.min.y,Je=le.isBox3?le.min.z:0;else{const yn=Math.pow(2,-K);Ue=Math.floor(Qt.width*yn),je=Math.floor(Qt.height*yn),D.isDataArrayTexture?Ve=Qt.depth:D.isData3DTexture?Ve=Math.floor(Qt.depth*yn):Ve=1,st=0,ot=0,Je=0}ae!==null?(mt=ae.x,Tt=ae.y,Yt=ae.z):(mt=0,Tt=0,Yt=0);const Et=$e.convert(Y.format),tt=$e.convert(Y.type);let Xt;Y.isData3DTexture?(ve.setTexture3D(Y,0),Xt=w.TEXTURE_3D):Y.isDataArrayTexture||Y.isCompressedArrayTexture?(ve.setTexture2DArray(Y,0),Xt=w.TEXTURE_2D_ARRAY):(ve.setTexture2D(Y,0),Xt=w.TEXTURE_2D),w.pixelStorei(w.UNPACK_FLIP_Y_WEBGL,Y.flipY),w.pixelStorei(w.UNPACK_PREMULTIPLY_ALPHA_WEBGL,Y.premultiplyAlpha),w.pixelStorei(w.UNPACK_ALIGNMENT,Y.unpackAlignment);const it=w.getParameter(w.UNPACK_ROW_LENGTH),dt=w.getParameter(w.UNPACK_IMAGE_HEIGHT),xt=w.getParameter(w.UNPACK_SKIP_PIXELS),Pt=w.getParameter(w.UNPACK_SKIP_ROWS),yi=w.getParameter(w.UNPACK_SKIP_IMAGES);w.pixelStorei(w.UNPACK_ROW_LENGTH,Qt.width),w.pixelStorei(w.UNPACK_IMAGE_HEIGHT,Qt.height),w.pixelStorei(w.UNPACK_SKIP_PIXELS,st),w.pixelStorei(w.UNPACK_SKIP_ROWS,ot),w.pixelStorei(w.UNPACK_SKIP_IMAGES,Je);const zt=D.isDataArrayTexture||D.isData3DTexture,xn=Y.isDataArrayTexture||Y.isData3DTexture;if(D.isDepthTexture){const yn=te.get(D),bn=te.get(Y),Rn=te.get(yn.__renderTarget),la=te.get(bn.__renderTarget);ie.bindFramebuffer(w.READ_FRAMEBUFFER,Rn.__webglFramebuffer),ie.bindFramebuffer(w.DRAW_FRAMEBUFFER,la.__webglFramebuffer);for(let Xi=0;Xi<Ve;Xi++)zt&&(w.framebufferTextureLayer(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,te.get(D).__webglTexture,K,Je+Xi),w.framebufferTextureLayer(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,te.get(Y).__webglTexture,Ee,Yt+Xi)),w.blitFramebuffer(st,ot,Ue,je,mt,Tt,Ue,je,w.DEPTH_BUFFER_BIT,w.NEAREST);ie.bindFramebuffer(w.READ_FRAMEBUFFER,null),ie.bindFramebuffer(w.DRAW_FRAMEBUFFER,null)}else if(K!==0||D.isRenderTargetTexture||te.has(D)){const yn=te.get(D),bn=te.get(Y);ie.bindFramebuffer(w.READ_FRAMEBUFFER,nu),ie.bindFramebuffer(w.DRAW_FRAMEBUFFER,gl);for(let Rn=0;Rn<Ve;Rn++)zt?w.framebufferTextureLayer(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,yn.__webglTexture,K,Je+Rn):w.framebufferTexture2D(w.READ_FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,yn.__webglTexture,K),xn?w.framebufferTextureLayer(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,bn.__webglTexture,Ee,Yt+Rn):w.framebufferTexture2D(w.DRAW_FRAMEBUFFER,w.COLOR_ATTACHMENT0,w.TEXTURE_2D,bn.__webglTexture,Ee),K!==0?w.blitFramebuffer(st,ot,Ue,je,mt,Tt,Ue,je,w.COLOR_BUFFER_BIT,w.NEAREST):xn?w.copyTexSubImage3D(Xt,Ee,mt,Tt,Yt+Rn,st,ot,Ue,je):w.copyTexSubImage2D(Xt,Ee,mt,Tt,st,ot,Ue,je);ie.bindFramebuffer(w.READ_FRAMEBUFFER,null),ie.bindFramebuffer(w.DRAW_FRAMEBUFFER,null)}else xn?D.isDataTexture||D.isData3DTexture?w.texSubImage3D(Xt,Ee,mt,Tt,Yt,Ue,je,Ve,Et,tt,Qt.data):Y.isCompressedArrayTexture?w.compressedTexSubImage3D(Xt,Ee,mt,Tt,Yt,Ue,je,Ve,Et,Qt.data):w.texSubImage3D(Xt,Ee,mt,Tt,Yt,Ue,je,Ve,Et,tt,Qt):D.isDataTexture?w.texSubImage2D(w.TEXTURE_2D,Ee,mt,Tt,Ue,je,Et,tt,Qt.data):D.isCompressedTexture?w.compressedTexSubImage2D(w.TEXTURE_2D,Ee,mt,Tt,Qt.width,Qt.height,Et,Qt.data):w.texSubImage2D(w.TEXTURE_2D,Ee,mt,Tt,Ue,je,Et,tt,Qt);w.pixelStorei(w.UNPACK_ROW_LENGTH,it),w.pixelStorei(w.UNPACK_IMAGE_HEIGHT,dt),w.pixelStorei(w.UNPACK_SKIP_PIXELS,xt),w.pixelStorei(w.UNPACK_SKIP_ROWS,Pt),w.pixelStorei(w.UNPACK_SKIP_IMAGES,yi),Ee===0&&Y.generateMipmaps&&w.generateMipmap(Xt),ie.unbindTexture()},this.initRenderTarget=function(D){te.get(D).__webglFramebuffer===void 0&&ve.setupRenderTarget(D)},this.initTexture=function(D){D.isCubeTexture?ve.setTextureCube(D,0):D.isData3DTexture?ve.setTexture3D(D,0):D.isDataArrayTexture||D.isCompressedArrayTexture?ve.setTexture2DArray(D,0):ve.setTexture2D(D,0),ie.unbindTexture()},this.resetState=function(){E=0,b=0,C=null,ie.reset(),W.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Ji}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=yt._getDrawingBufferColorSpace(e),t.unpackColorSpace=yt._getUnpackColorSpace()}}class Zs{static idGen=0;constructor(e,t){let n,s;this.promise=new Promise((c,u)=>{n=c,s=u});const r=n.bind(this),o=s.bind(this),a=(...c)=>{r(...c)},l=c=>{o(c)};e(a.bind(this),l.bind(this)),this.abortHandler=t,this.id=Zs.idGen++}then(e){return new Zs((t,n)=>{this.promise=this.promise.then((...s)=>{const r=e(...s);r instanceof Promise||r instanceof Zs?r.then((...o)=>{t(...o)}):t(r)}).catch(s=>{n(s)})},this.abortHandler)}catch(e){return new Zs(t=>{this.promise=this.promise.then((...n)=>{t(...n)}).catch(e)},this.abortHandler)}abort(e){this.abortHandler&&this.abortHandler(e)}}class nx extends Error{constructor(e){super(e)}}(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){i[0]=t;const n=e[0];let s=n>>16&32768,r=n>>12&2047;const o=n>>23&255;return o<103?s:o>142?(s|=31744,s|=(o==255?0:1)&&n&8388607,s):o<113?(r|=2048,s|=(r>>114-o)+(r>>113-o&1),s):(s|=o-112<<10|r>>1,s+=r&1,s)}})();const Yu=(function(){const i=new Float32Array(1),e=new Int32Array(i.buffer);return function(t){return i[0]=t,e[0]}})(),_1=function(i,e){return i[e]+(i[e+1]<<8)+(i[e+2]<<16)+(i[e+3]<<24)},Qc=function(i,e,t=!0,n){const s=new AbortController,r=s.signal;let o=!1;const a=u=>{s.abort(u),o=!0};let l=!1;const c=(u,f,d,h)=>{e&&!l&&(e(u,f,d,h),u===100&&(l=!0))};return new Zs((u,f)=>{const d={signal:r};n&&(d.headers=n),fetch(i,d).then(async h=>{if(!h.ok){const v=await h.text();f(new Error(`Fetch failed: ${h.status} ${h.statusText} ${v}`));return}const x=h.body.getReader();let p=0,g=h.headers.get("Content-Length"),m=g?parseInt(g):void 0;const _=[];for(;!o;)try{const{value:v,done:A}=await x.read();if(A){if(c(100,"100%",v,m),t){const M=new Blob(_).arrayBuffer();u(M)}else u();break}p+=v.length;let S,y;m!==void 0&&(S=p/m*100,y=`${S.toFixed(2)}%`),t&&_.push(v),c(S,y,v,m)}catch(v){f(v);return}}).catch(h=>{f(new nx(h))})},a)},jt=function(i,e,t){return Math.max(Math.min(i,t),e)},xo=function(){return performance.now()/1e3},yo=i=>{if(i.geometry&&(i.geometry.dispose(),i.geometry=null),i.material&&(i.material.dispose(),i.material=null),i.children)for(let e of i.children)yo(e)},pi=(i,e)=>new Promise(t=>{window.setTimeout(()=>{t(i?i():void 0)},e?1:50)}),Fo=(i=0)=>{let e=0;if(i===1)e=9;else if(i===2)e=24;else if(i===3)e=45;else if(i>3)throw new Error("getSphericalHarmonicsComponentCountForDegree() -> Invalid spherical harmonics degree");return e},oh=()=>{let i,e;return{promise:new Promise((n,s)=>{i=n,e=s}),resolve:i,reject:e}},Qu=i=>{let e,t;return i||(i=()=>{}),{promise:new Zs((s,r)=>{e=s,t=r},i),resolve:e,reject:t}};class v1{constructor(e,t,n){this.major=e,this.minor=t,this.patch=n}toString(){return`${this.major}_${this.minor}_${this.patch}`}}function ah(){const i=navigator.userAgent;return i.indexOf("iPhone")>0||i.indexOf("iPad")>0}function ix(){if(ah()){const i=navigator.userAgent.match(/OS (\d+)_(\d+)_?(\d+)?/);return new v1(parseInt(i[1]||0,10),parseInt(i[2]||0,10),parseInt(i[3]||0,10))}else return null}const A1=14;class De{static OFFSET={X:0,Y:1,Z:2,SCALE0:3,SCALE1:4,SCALE2:5,ROTATION0:6,ROTATION1:7,ROTATION2:8,ROTATION3:9,FDC0:10,FDC1:11,FDC2:12,OPACITY:13,FRC0:14,FRC1:15,FRC2:16,FRC3:17,FRC4:18,FRC5:19,FRC6:20,FRC7:21,FRC8:22,FRC9:23,FRC10:24,FRC11:25,FRC12:26,FRC13:27,FRC14:28,FRC15:29,FRC16:30,FRC17:31,FRC18:32,FRC19:33,FRC20:34,FRC21:35,FRC22:36,FRC23:37};constructor(e=0){this.sphericalHarmonicsDegree=e,this.sphericalHarmonicsCount=Fo(this.sphericalHarmonicsDegree),this.componentCount=this.sphericalHarmonicsCount+A1,this.defaultSphericalHarmonics=new Array(this.sphericalHarmonicsCount).fill(0),this.splats=[],this.splatCount=0}static createSplat(e=0){const t=[0,0,0,1,1,1,1,0,0,0,0,0,0,0];let n=Fo(e);for(let s=0;s<n;s++)t.push(0);return t}addSplat(e){this.splats.push(e),this.splatCount++}getSplat(e){return this.splats[e]}addDefaultSplat(){const e=De.createSplat(this.sphericalHarmonicsDegree);return this.addSplat(e),e}addSplatFromComonents(e,t,n,s,r,o,a,l,c,u,f,d,h,x,...p){const g=[e,t,n,s,r,o,a,l,c,u,f,d,h,x,...this.defaultSphericalHarmonics];for(let m=0;m<p.length&&m<this.sphericalHarmonicsCount;m++)g[m]=p[m];return this.addSplat(g),g}addSplatFromArray(e,t){const n=e.splats[t],s=De.createSplat(this.sphericalHarmonicsDegree);for(let r=0;r<this.componentCount&&r<n.length;r++)s[r]=n[r];this.addSplat(s)}}class Ot{static DefaultSplatSortDistanceMapPrecision=16;static MemoryPageSize=65536;static BytesPerFloat=4;static BytesPerInt=4;static MaxScenes=32;static ProgressiveLoadSectionSize=262144;static ProgressiveLoadSectionDelayDuration=15;static SphericalHarmonics8BitCompressionRange=3}const S1=Ot.SphericalHarmonics8BitCompressionRange,qs=S1/2,pn=Za.toHalfFloat.bind(Za),lh=Za.fromHalfFloat.bind(Za),Kt=(i,e,t=!1,n,s)=>{if(e===0)return i;if(e===1||e===2&&!t)return Za.fromHalfFloat(i);if(e===2)return ch(i,n,s)},Pa=(i,e,t)=>{i=jt(i,e,t);const n=t-e;return jt(Math.floor((i-e)/n*255),0,255)},ch=(i,e,t)=>{const n=t-e;return i/255*n+e},sx=(i,e,t)=>Pa(lh(i,e,t)),y1=(i,e,t)=>pn(ch(i,e,t)),wt=(i,e,t,n=!1)=>t===0?i.getFloat32(e*4,!0):t===1||t===2&&!n?i.getUint16(e*2,!0):i.getUint8(e,!0),b1=(function(){const i=e=>e;return function(e,t,n,s=!1){if(t===n)return e;let r=i;return t===2&&s?n===1?r=y1:n==0&&(r=ch):t===2||t===1?n===0?r=lh:n==2&&(s?r=sx:r=i):t===0&&(n===1?r=pn:n==2&&(s?r=Pa:r=pn)),r(e)}})(),_o=(i,e,t,n,s=0)=>{const r=new Uint8Array(i,e),o=new Uint8Array(t,n);for(let a=0;a<s;a++)o[a]=r[a]};class Z{static CurrentMajorVersion=0;static CurrentMinorVersion=1;static CenterComponentCount=3;static ScaleComponentCount=3;static RotationComponentCount=4;static ColorComponentCount=4;static CovarianceComponentCount=6;static SplatScaleOffsetFloat=3;static SplatRotationOffsetFloat=6;static CompressionLevels={0:{BytesPerCenter:12,BytesPerScale:12,BytesPerRotation:16,BytesPerColor:4,ScaleOffsetBytes:12,RotationffsetBytes:24,ColorOffsetBytes:40,SphericalHarmonicsOffsetBytes:44,ScaleRange:1,BytesPerSphericalHarmonicsComponent:4,SphericalHarmonicsOffsetFloat:11,SphericalHarmonicsDegrees:{0:{BytesPerSplat:44},1:{BytesPerSplat:80},2:{BytesPerSplat:140}}},1:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:2,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:42},2:{BytesPerSplat:72}}},2:{BytesPerCenter:6,BytesPerScale:6,BytesPerRotation:8,BytesPerColor:4,ScaleOffsetBytes:6,RotationffsetBytes:12,ColorOffsetBytes:20,SphericalHarmonicsOffsetBytes:24,ScaleRange:32767,BytesPerSphericalHarmonicsComponent:1,SphericalHarmonicsOffsetFloat:12,SphericalHarmonicsDegrees:{0:{BytesPerSplat:24},1:{BytesPerSplat:33},2:{BytesPerSplat:48}}}};static CovarianceSizeFloats=6;static HeaderSizeBytes=4096;static SectionHeaderSizeBytes=1024;static BucketStorageSizeBytes=12;static BucketStorageSizeFloats=3;static BucketBlockSize=5;static BucketSize=256;constructor(e,t=!0){this.constructFromBuffer(e,t)}getSplatCount(){return this.splatCount}getMaxSplatCount(){return this.maxSplatCount}getMinSphericalHarmonicsDegree(){let e=0;for(let t=0;t<this.sections.length;t++){const n=this.sections[t];(t===0||n.sphericalHarmonicsDegree<e)&&(e=n.sphericalHarmonicsDegree)}return e}getBucketIndex(e,t){let n;const s=e.fullBucketCount*e.bucketSize;if(t<s)n=Math.floor(t/e.bucketSize);else{let r=s;n=e.fullBucketCount;let o=0;for(;r<e.splatCount;){let a=e.partiallyFilledBucketLengths[o];if(t>=r&&t<r+a)break;r+=a,n++,o++}}return n}getSplatCenter(e,t,n){const s=this.globalSplatIndexToSectionMap[e],r=this.sections[s],o=e-r.splatCountOffset,a=r.bytesPerSplat*o,l=new DataView(this.bufferData,r.dataBase+a),c=wt(l,0,this.compressionLevel),u=wt(l,1,this.compressionLevel),f=wt(l,2,this.compressionLevel);if(this.compressionLevel>=1){const h=this.getBucketIndex(r,o)*Z.BucketStorageSizeFloats,x=r.compressionScaleFactor,p=r.compressionScaleRange;t.x=(c-p)*x+r.bucketArray[h],t.y=(u-p)*x+r.bucketArray[h+1],t.z=(f-p)*x+r.bucketArray[h+2]}else t.x=c,t.y=u,t.z=f;n&&t.applyMatrix4(n)}getSplatScaleAndRotation=(function(){const e=new rt,t=new rt,n=new rt,s=new U,r=new U,o=new Gt;return function(a,l,c,u,f){const d=this.globalSplatIndexToSectionMap[a],h=this.sections[d],x=a-h.splatCountOffset,p=h.bytesPerSplat*x+Z.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,g=new DataView(this.bufferData,h.dataBase+p);r.set(Kt(wt(g,0,this.compressionLevel),this.compressionLevel),Kt(wt(g,1,this.compressionLevel),this.compressionLevel),Kt(wt(g,2,this.compressionLevel),this.compressionLevel)),f&&(f.x!==void 0&&(r.x=f.x),f.y!==void 0&&(r.y=f.y),f.z!==void 0&&(r.z=f.z)),o.set(Kt(wt(g,4,this.compressionLevel),this.compressionLevel),Kt(wt(g,5,this.compressionLevel),this.compressionLevel),Kt(wt(g,6,this.compressionLevel),this.compressionLevel),Kt(wt(g,3,this.compressionLevel),this.compressionLevel)),u?(e.makeScale(r.x,r.y,r.z),t.makeRotationFromQuaternion(o),n.copy(e).multiply(t).multiply(u),n.decompose(s,c,l)):(l.copy(r),c.copy(o))}})();getSplatColor(e,t){const n=this.globalSplatIndexToSectionMap[e],s=this.sections[n],r=e-s.splatCountOffset,o=s.bytesPerSplat*r+Z.CompressionLevels[this.compressionLevel].ColorOffsetBytes,a=new Uint8Array(this.bufferData,s.dataBase+o,4);t.set(a[0],a[1],a[2],a[3])}fillSplatCenterArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);const a=new U;for(let l=n;l<=s;l++){const c=this.globalSplatIndexToSectionMap[l],u=this.sections[c],f=l-u.splatCountOffset,d=(l-n+r)*Z.CenterComponentCount,h=u.bytesPerSplat*f,x=new DataView(this.bufferData,u.dataBase+h),p=wt(x,0,this.compressionLevel),g=wt(x,1,this.compressionLevel),m=wt(x,2,this.compressionLevel);if(this.compressionLevel>=1){const v=this.getBucketIndex(u,f)*Z.BucketStorageSizeFloats,A=u.compressionScaleFactor,S=u.compressionScaleRange;a.x=(p-S)*A+u.bucketArray[v],a.y=(g-S)*A+u.bucketArray[v+1],a.z=(m-S)*A+u.bucketArray[v+2]}else a.x=p,a.y=g,a.z=m;t&&a.applyMatrix4(t),e[d]=a.x,e[d+1]=a.y,e[d+2]=a.z}}fillSplatScaleRotationArray=(function(){const e=new rt,t=new rt,n=new rt,s=new U,r=new Gt,o=new U,a=l=>{const c=l.w<0?-1:1;l.x*=c,l.y*=c,l.z*=c,l.w*=c};return function(l,c,u,f,d,h,x,p){const g=this.splatCount;f=f||0,d=d||g-1,h===void 0&&(h=f);const m=(_,v)=>b1(_,v,x);for(let _=f;_<=d;_++){const v=this.globalSplatIndexToSectionMap[_],A=this.sections[v],S=_-A.splatCountOffset,y=A.bytesPerSplat*S+Z.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,M=(_-f+h)*Z.ScaleComponentCount,E=(_-f+h)*Z.RotationComponentCount,b=new DataView(this.bufferData,A.dataBase+y),C=p&&p.x!==void 0?p.x:wt(b,0,this.compressionLevel),P=p&&p.y!==void 0?p.y:wt(b,1,this.compressionLevel),B=p&&p.z!==void 0?p.z:wt(b,2,this.compressionLevel),N=wt(b,3,this.compressionLevel),k=wt(b,4,this.compressionLevel),V=wt(b,5,this.compressionLevel),F=wt(b,6,this.compressionLevel);s.set(Kt(C,this.compressionLevel),Kt(P,this.compressionLevel),Kt(B,this.compressionLevel)),r.set(Kt(k,this.compressionLevel),Kt(V,this.compressionLevel),Kt(F,this.compressionLevel),Kt(N,this.compressionLevel)).normalize(),u&&(o.set(0,0,0),e.makeScale(s.x,s.y,s.z),t.makeRotationFromQuaternion(r),n.identity().premultiply(e).premultiply(t),n.premultiply(u),n.decompose(o,r,s),r.normalize()),a(r),l&&(l[M]=m(s.x,0),l[M+1]=m(s.y,0),l[M+2]=m(s.z,0)),c&&(c[E]=m(r.x,0),c[E+1]=m(r.y,0),c[E+2]=m(r.z,0),c[E+3]=m(r.w,0))}}})();static computeCovariance=(function(){const e=new rt,t=new at,n=new at,s=new at,r=new at,o=new at,a=new at;return function(l,c,u,f,d=0,h){e.makeScale(l.x,l.y,l.z),t.setFromMatrix4(e),e.makeRotationFromQuaternion(c),n.setFromMatrix4(e),s.copy(n).multiply(t),r.copy(s).transpose().premultiply(s),u&&(o.setFromMatrix4(u),a.copy(o).transpose(),r.multiply(a),r.premultiply(o)),h>=1?(f[d]=pn(r.elements[0]),f[d+1]=pn(r.elements[3]),f[d+2]=pn(r.elements[6]),f[d+3]=pn(r.elements[4]),f[d+4]=pn(r.elements[7]),f[d+5]=pn(r.elements[8])):(f[d]=r.elements[0],f[d+1]=r.elements[3],f[d+2]=r.elements[6],f[d+3]=r.elements[4],f[d+4]=r.elements[7],f[d+5]=r.elements[8])}})();fillSplatCovarianceArray(e,t,n,s,r,o){const a=this.splatCount,l=new U,c=new Gt;n=n||0,s=s||a-1,r===void 0&&(r=n);for(let u=n;u<=s;u++){const f=this.globalSplatIndexToSectionMap[u],d=this.sections[f],h=u-d.splatCountOffset,x=(u-n+r)*Z.CovarianceComponentCount,p=d.bytesPerSplat*h+Z.CompressionLevels[this.compressionLevel].ScaleOffsetBytes,g=new DataView(this.bufferData,d.dataBase+p);l.set(Kt(wt(g,0,this.compressionLevel),this.compressionLevel),Kt(wt(g,1,this.compressionLevel),this.compressionLevel),Kt(wt(g,2,this.compressionLevel),this.compressionLevel)),c.set(Kt(wt(g,4,this.compressionLevel),this.compressionLevel),Kt(wt(g,5,this.compressionLevel),this.compressionLevel),Kt(wt(g,6,this.compressionLevel),this.compressionLevel),Kt(wt(g,3,this.compressionLevel),this.compressionLevel)),Z.computeCovariance(l,c,t,e,x,o)}}fillSplatColorArray(e,t,n,s,r){const o=this.splatCount;n=n||0,s=s||o-1,r===void 0&&(r=n);for(let a=n;a<=s;a++){const l=this.globalSplatIndexToSectionMap[a],c=this.sections[l],u=a-c.splatCountOffset,f=(a-n+r)*Z.ColorComponentCount,d=c.bytesPerSplat*u+Z.CompressionLevels[this.compressionLevel].ColorOffsetBytes,h=new Uint8Array(this.bufferData,c.dataBase+d);let x=h[3];x=x>=t?x:0,e[f]=h[0],e[f+1]=h[1],e[f+2]=h[2],e[f+3]=x}}fillSphericalHarmonicsArray=(function(){for(let k=0;k<15;k++)new U;const e=new at,t=new rt,n=new U,s=new U,r=new Gt,o=[],a=[],l=[],c=[],u=[],f=[],d=[],h=[],x=[],p=[],g=[],m=[],_=[],v=[],A=[],S=[],y=[],M=[],E=k=>k,b=(k,V,F,q)=>{k[0]=V,k[1]=F,k[2]=q},C=(k,V,F,q,G)=>{k[0]=wt(V,q,G,!0),k[1]=wt(V,q+F,G,!0),k[2]=wt(V,q+F+F,G,!0)},P=(k,V)=>{V[0]=k[0],V[1]=k[1],V[2]=k[2]},B=(k,V,F,q)=>{V[F]=q(k[0]),V[F+1]=q(k[1]),V[F+2]=q(k[2])},N=(k,V,F,q,G)=>(V[0]=Kt(k[0],F,!0,q,G),V[1]=Kt(k[1],F,!0,q,G),V[2]=Kt(k[2],F,!0,q,G),V);return function(k,V,F,q,G,j,ce){const ye=this.splatCount;q=q||0,G=G||ye-1,j===void 0&&(j=q),F&&V>=1&&(t.copy(F),t.decompose(n,r,s),r.normalize(),t.makeRotationFromQuaternion(r),e.setFromMatrix4(t),b(o,e.elements[4],-e.elements[7],e.elements[1]),b(a,-e.elements[5],e.elements[8],-e.elements[2]),b(l,e.elements[3],-e.elements[6],e.elements[0]));const Ie=de=>sx(de,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff),Ne=de=>Pa(de,this.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff);for(let de=q;de<=G;de++){const qe=this.globalSplatIndexToSectionMap[de],ee=this.sections[qe];V=Math.min(V,ee.sphericalHarmonicsDegree);const ue=Fo(V),Me=de-ee.splatCountOffset,Ge=ee.bytesPerSplat*Me+Z.CompressionLevels[this.compressionLevel].SphericalHarmonicsOffsetBytes,Ce=new DataView(this.bufferData,ee.dataBase+Ge),Ye=(de-q+j)*ue;let O=F?0:this.compressionLevel,z=E;O!==ce&&(O===1?ce===0?z=lh:ce==2&&(z=Ie):O===0&&(ce===1?z=pn:ce==2&&(z=Ne)));const Q=this.minSphericalHarmonicsCoeff,w=this.maxSphericalHarmonicsCoeff;V>=1&&(C(x,Ce,3,0,this.compressionLevel),C(p,Ce,3,1,this.compressionLevel),C(g,Ce,3,2,this.compressionLevel),F?(N(x,x,this.compressionLevel,Q,w),N(p,p,this.compressionLevel,Q,w),N(g,g,this.compressionLevel,Q,w),Z.rotateSphericalHarmonics3(x,p,g,o,a,l,v,A,S)):(P(x,v),P(p,A),P(g,S)),B(v,k,Ye,z),B(A,k,Ye+3,z),B(S,k,Ye+6,z),V>=2&&(C(x,Ce,5,9,this.compressionLevel),C(p,Ce,5,10,this.compressionLevel),C(g,Ce,5,11,this.compressionLevel),C(m,Ce,5,12,this.compressionLevel),C(_,Ce,5,13,this.compressionLevel),F?(N(x,x,this.compressionLevel,Q,w),N(p,p,this.compressionLevel,Q,w),N(g,g,this.compressionLevel,Q,w),N(m,m,this.compressionLevel,Q,w),N(_,_,this.compressionLevel,Q,w),Z.rotateSphericalHarmonics5(x,p,g,m,_,o,a,l,c,u,f,d,h,v,A,S,y,M)):(P(x,v),P(p,A),P(g,S),P(m,y),P(_,M)),B(v,k,Ye+9,z),B(A,k,Ye+12,z),B(S,k,Ye+15,z),B(y,k,Ye+18,z),B(M,k,Ye+21,z)))}}})();static dot3=(e,t,n,s,r)=>{r[0]=r[1]=r[2]=0;const o=s[0],a=s[1],l=s[2];Z.addInto3(e[0]*o,e[1]*o,e[2]*o,r),Z.addInto3(t[0]*a,t[1]*a,t[2]*a,r),Z.addInto3(n[0]*l,n[1]*l,n[2]*l,r)};static addInto3=(e,t,n,s)=>{s[0]=s[0]+e,s[1]=s[1]+t,s[2]=s[2]+n};static dot5=(e,t,n,s,r,o,a)=>{a[0]=a[1]=a[2]=0;const l=o[0],c=o[1],u=o[2],f=o[3],d=o[4];Z.addInto3(e[0]*l,e[1]*l,e[2]*l,a),Z.addInto3(t[0]*c,t[1]*c,t[2]*c,a),Z.addInto3(n[0]*u,n[1]*u,n[2]*u,a),Z.addInto3(s[0]*f,s[1]*f,s[2]*f,a),Z.addInto3(r[0]*d,r[1]*d,r[2]*d,a)};static rotateSphericalHarmonics3=(e,t,n,s,r,o,a,l,c)=>{Z.dot3(e,t,n,s,a),Z.dot3(e,t,n,r,l),Z.dot3(e,t,n,o,c)};static rotateSphericalHarmonics5=(e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g,m,_)=>{const v=Math.sqrt(.25),A=Math.sqrt(3/4),S=Math.sqrt(1/3),y=Math.sqrt(4/3),M=Math.sqrt(1/12);c[0]=v*(l[2]*o[0]+l[0]*o[2]+(o[2]*l[0]+o[0]*l[2])),c[1]=l[1]*o[0]+o[1]*l[0],c[2]=A*(l[1]*o[1]+o[1]*l[1]),c[3]=l[1]*o[2]+o[1]*l[2],c[4]=v*(l[2]*o[2]-l[0]*o[0]+(o[2]*l[2]-o[0]*l[0])),Z.dot5(e,t,n,s,r,c,x),u[0]=v*(a[2]*o[0]+a[0]*o[2]+(o[2]*a[0]+o[0]*a[2])),u[1]=a[1]*o[0]+o[1]*a[0],u[2]=A*(a[1]*o[1]+o[1]*a[1]),u[3]=a[1]*o[2]+o[1]*a[2],u[4]=v*(a[2]*o[2]-a[0]*o[0]+(o[2]*a[2]-o[0]*a[0])),Z.dot5(e,t,n,s,r,u,p),f[0]=S*(a[2]*a[0]+a[0]*a[2])+-M*(l[2]*l[0]+l[0]*l[2]+(o[2]*o[0]+o[0]*o[2])),f[1]=y*a[1]*a[0]+-S*(l[1]*l[0]+o[1]*o[0]),f[2]=a[1]*a[1]+-v*(l[1]*l[1]+o[1]*o[1]),f[3]=y*a[1]*a[2]+-S*(l[1]*l[2]+o[1]*o[2]),f[4]=S*(a[2]*a[2]-a[0]*a[0])+-M*(l[2]*l[2]-l[0]*l[0]+(o[2]*o[2]-o[0]*o[0])),Z.dot5(e,t,n,s,r,f,g),d[0]=v*(a[2]*l[0]+a[0]*l[2]+(l[2]*a[0]+l[0]*a[2])),d[1]=a[1]*l[0]+l[1]*a[0],d[2]=A*(a[1]*l[1]+l[1]*a[1]),d[3]=a[1]*l[2]+l[1]*a[2],d[4]=v*(a[2]*l[2]-a[0]*l[0]+(l[2]*a[2]-l[0]*a[0])),Z.dot5(e,t,n,s,r,d,m),h[0]=v*(l[2]*l[0]+l[0]*l[2]-(o[2]*o[0]+o[0]*o[2])),h[1]=l[1]*l[0]-o[1]*o[0],h[2]=A*(l[1]*l[1]-o[1]*o[1]),h[3]=l[1]*l[2]-o[1]*o[2],h[4]=v*(l[2]*l[2]-l[0]*l[0]-(o[2]*o[2]-o[0]*o[0])),Z.dot5(e,t,n,s,r,h,_)};static parseHeader(e){const t=new Uint8Array(e,0,Z.HeaderSizeBytes),n=new Uint16Array(e,0,Z.HeaderSizeBytes/2),s=new Uint32Array(e,0,Z.HeaderSizeBytes/4),r=new Float32Array(e,0,Z.HeaderSizeBytes/4),o=t[0],a=t[1],l=s[1],c=s[2],u=s[3],f=s[4],d=n[10],h=new U(r[6],r[7],r[8]),x=r[9]||-qs,p=r[10]||qs;return{versionMajor:o,versionMinor:a,maxSectionCount:l,sectionCount:c,maxSplatCount:u,splatCount:f,compressionLevel:d,sceneCenter:h,minSphericalHarmonicsCoeff:x,maxSphericalHarmonicsCoeff:p}}static writeHeaderCountsToBuffer(e,t,n){const s=new Uint32Array(n,0,Z.HeaderSizeBytes/4);s[2]=e,s[4]=t}static writeHeaderToBuffer(e,t){const n=new Uint8Array(t,0,Z.HeaderSizeBytes),s=new Uint16Array(t,0,Z.HeaderSizeBytes/2),r=new Uint32Array(t,0,Z.HeaderSizeBytes/4),o=new Float32Array(t,0,Z.HeaderSizeBytes/4);n[0]=e.versionMajor,n[1]=e.versionMinor,n[2]=0,n[3]=0,r[1]=e.maxSectionCount,r[2]=e.sectionCount,r[3]=e.maxSplatCount,r[4]=e.splatCount,s[10]=e.compressionLevel,o[6]=e.sceneCenter.x,o[7]=e.sceneCenter.y,o[8]=e.sceneCenter.z,o[9]=e.minSphericalHarmonicsCoeff||-qs,o[10]=e.maxSphericalHarmonicsCoeff||qs}static parseSectionHeaders(e,t,n=0,s){const r=e.compressionLevel,o=e.maxSectionCount,a=new Uint16Array(t,n,o*Z.SectionHeaderSizeBytes/2),l=new Uint32Array(t,n,o*Z.SectionHeaderSizeBytes/4),c=new Float32Array(t,n,o*Z.SectionHeaderSizeBytes/4),u=[];let f=0,d=f/2,h=f/4,x=Z.HeaderSizeBytes+e.maxSectionCount*Z.SectionHeaderSizeBytes,p=0;for(let g=0;g<o;g++){const m=l[h+1],_=l[h+2],v=l[h+3],A=c[h+4],S=A/2,y=a[d+10],M=l[h+6]||Z.CompressionLevels[r].ScaleRange,E=l[h+8],b=l[h+9],C=b*4,P=y*v+C,B=a[d+20],{bytesPerSplat:N}=Z.calculateComponentStorage(r,B),k=N*m,V=k+P,F={bytesPerSplat:N,splatCountOffset:p,splatCount:s?m:0,maxSplatCount:m,bucketSize:_,bucketCount:v,bucketBlockSize:A,halfBucketBlockSize:S,bucketStorageSizeBytes:y,bucketsStorageSizeBytes:P,splatDataStorageSizeBytes:k,storageSizeBytes:V,compressionScaleRange:M,compressionScaleFactor:S/M,base:x,bucketsBase:x+C,dataBase:x+P,fullBucketCount:E,partiallyFilledBucketCount:b,sphericalHarmonicsDegree:B};u[g]=F,x+=V,f+=Z.SectionHeaderSizeBytes,d=f/2,h=f/4,p+=m}return u}static writeSectionHeaderToBuffer(e,t,n,s=0){const r=new Uint16Array(n,s,Z.SectionHeaderSizeBytes/2),o=new Uint32Array(n,s,Z.SectionHeaderSizeBytes/4),a=new Float32Array(n,s,Z.SectionHeaderSizeBytes/4);o[0]=e.splatCount,o[1]=e.maxSplatCount,o[2]=t>=1?e.bucketSize:0,o[3]=t>=1?e.bucketCount:0,a[4]=t>=1?e.bucketBlockSize:0,r[10]=t>=1?Z.BucketStorageSizeBytes:0,o[6]=t>=1?e.compressionScaleRange:0,o[7]=e.storageSizeBytes,o[8]=t>=1?e.fullBucketCount:0,o[9]=t>=1?e.partiallyFilledBucketCount:0,r[20]=e.sphericalHarmonicsDegree}static writeSectionHeaderSplatCountToBuffer(e,t,n=0){const s=new Uint32Array(t,n,Z.SectionHeaderSizeBytes/4);s[0]=e}constructFromBuffer(e,t){this.bufferData=e,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSectionMap=[];const n=Z.parseHeader(this.bufferData);this.versionMajor=n.versionMajor,this.versionMinor=n.versionMinor,this.maxSectionCount=n.maxSectionCount,this.sectionCount=t?n.maxSectionCount:0,this.maxSplatCount=n.maxSplatCount,this.splatCount=t?n.maxSplatCount:0,this.compressionLevel=n.compressionLevel,this.sceneCenter=new U().copy(n.sceneCenter),this.minSphericalHarmonicsCoeff=n.minSphericalHarmonicsCoeff,this.maxSphericalHarmonicsCoeff=n.maxSphericalHarmonicsCoeff,this.sections=Z.parseSectionHeaders(n,this.bufferData,Z.HeaderSizeBytes,t),this.linkBufferArrays(),this.buildMaps()}static calculateComponentStorage(e,t){const n=Z.CompressionLevels[e].BytesPerCenter,s=Z.CompressionLevels[e].BytesPerScale,r=Z.CompressionLevels[e].BytesPerRotation,o=Z.CompressionLevels[e].BytesPerColor,a=Fo(t),l=Z.CompressionLevels[e].BytesPerSphericalHarmonicsComponent*a,c=n+s+r+o+l;return{bytesPerCenter:n,bytesPerScale:s,bytesPerRotation:r,bytesPerColor:o,sphericalHarmonicsComponentsPerSplat:a,sphericalHarmonicsBytesPerSplat:l,bytesPerSplat:c}}linkBufferArrays(){for(let e=0;e<this.maxSectionCount;e++){const t=this.sections[e];t.bucketArray=new Float32Array(this.bufferData,t.bucketsBase,t.bucketCount*Z.BucketStorageSizeFloats),t.partiallyFilledBucketCount>0&&(t.partiallyFilledBucketLengths=new Uint32Array(this.bufferData,t.base,t.partiallyFilledBucketCount))}}buildMaps(){let e=0;for(let t=0;t<this.maxSectionCount;t++){const n=this.sections[t];for(let s=0;s<n.maxSplatCount;s++){const r=e+s;this.globalSplatIndexToLocalSplatIndexMap[r]=s,this.globalSplatIndexToSectionMap[r]=t}e+=n.maxSplatCount}}updateLoadedCounts(e,t){Z.writeHeaderCountsToBuffer(e,t,this.bufferData),this.sectionCount=e,this.splatCount=t}updateSectionLoadedCounts(e,t){const n=Z.HeaderSizeBytes+Z.SectionHeaderSizeBytes*e;Z.writeSectionHeaderSplatCountToBuffer(t,this.bufferData,n),this.sections[e].splatCount=t}static writeSplatDataToSectionBuffer=(function(){const e=new ArrayBuffer(12),t=new ArrayBuffer(12),n=new ArrayBuffer(16),s=new ArrayBuffer(4),r=new ArrayBuffer(256),o=new Gt,a=new U,l=new U,{X:c,Y:u,Z:f,SCALE0:d,SCALE1:h,SCALE2:x,ROTATION0:p,ROTATION1:g,ROTATION2:m,ROTATION3:_,FDC0:v,FDC1:A,FDC2:S,OPACITY:y,FRC0:M,FRC9:E}=De.OFFSET,b=(C,P,B)=>{const N=B*2+1;return C=Math.round(C*P)+B,jt(C,0,N)};return function(C,P,B,N,k,V,F,q,G=-qs,j=qs){const ce=Fo(k),ye=Z.CompressionLevels[N].BytesPerCenter,Ie=Z.CompressionLevels[N].BytesPerScale,Ne=Z.CompressionLevels[N].BytesPerRotation,de=Z.CompressionLevels[N].BytesPerColor,qe=B,ee=qe+ye,ue=ee+Ie,Me=ue+Ne,Ge=Me+de;if(C[p]!==void 0?(o.set(C[p],C[g],C[m],C[_]),o.normalize()):o.set(1,0,0,0),C[d]!==void 0?a.set(C[d]||0,C[h]||0,C[x]||0):a.set(0,0,0),N===0){const Ye=new Float32Array(P,qe,Z.CenterComponentCount),O=new Float32Array(P,ue,Z.RotationComponentCount),z=new Float32Array(P,ee,Z.ScaleComponentCount);if(O.set([o.x,o.y,o.z,o.w]),z.set([a.x,a.y,a.z]),Ye.set([C[c],C[u],C[f]]),k>0){const Q=new Float32Array(P,Ge,ce);if(k>=1){for(let w=0;w<9;w++)Q[w]=C[M+w]||0;if(k>=2)for(let w=0;w<15;w++)Q[w+9]=C[E+w]||0}}}else{const Ye=new Uint16Array(e,0,Z.CenterComponentCount),O=new Uint16Array(n,0,Z.RotationComponentCount),z=new Uint16Array(t,0,Z.ScaleComponentCount);if(O.set([pn(o.x),pn(o.y),pn(o.z),pn(o.w)]),z.set([pn(a.x),pn(a.y),pn(a.z)]),l.set(C[c],C[u],C[f]).sub(V),l.x=b(l.x,F,q),l.y=b(l.y,F,q),l.z=b(l.z,F,q),Ye.set([l.x,l.y,l.z]),k>0){const Q=N===1?Uint16Array:Uint8Array,w=N===1?2:1,ne=new Q(r,0,ce);if(k>=1){for(let he=0;he<9;he++){const ie=C[M+he]||0;ne[he]=N===1?pn(ie):Pa(ie,G,j)}const oe=9*w;if(_o(ne.buffer,0,P,Ge,oe),k>=2){for(let he=0;he<15;he++){const ie=C[E+he]||0;ne[he+9]=N===1?pn(ie):Pa(ie,G,j)}_o(ne.buffer,oe,P,Ge+oe,15*w)}}}_o(Ye.buffer,0,P,qe,6),_o(z.buffer,0,P,ee,6),_o(O.buffer,0,P,ue,8)}const Ce=new Uint8ClampedArray(s,0,4);Ce.set([C[v]||0,C[A]||0,C[S]||0]),Ce[3]=C[y]||0,_o(Ce.buffer,0,P,Me,4)}})();static generateFromUncompressedSplatArrays(e,t,n,s,r,o,a=[]){let l=0;for(let S=0;S<e.length;S++){const y=e[S];l=Math.max(y.sphericalHarmonicsDegree,l)}let c,u;for(let S=0;S<e.length;S++){const y=e[S];for(let M=0;M<y.splats.length;M++){const E=y.splats[M];for(let b=De.OFFSET.FRC0;b<De.OFFSET.FRC23&&b<E.length;b++)(!c||E[b]<c)&&(c=E[b]),(!u||E[b]>u)&&(u=E[b])}}c=c||-qs,u=u||qs;const{bytesPerSplat:f}=Z.calculateComponentStorage(n,l),d=Z.CompressionLevels[n].ScaleRange,h=[],x=[];let p=0;for(let S=0;S<e.length;S++){const y=e[S],M=new De(l);for(let qe=0;qe<y.splatCount;qe++){const ee=y.splats[qe];(ee[De.OFFSET.OPACITY]||0)>=t&&M.addSplat(ee)}const E=a[S]||{},b=(E.blockSizeFactor||1)*(r||Z.BucketBlockSize),C=Math.ceil((E.bucketSizeFactor||1)*(o||Z.BucketSize)),P=Z.computeBucketsForUncompressedSplatArray(M,b,C),B=P.fullBuckets.length,N=P.partiallyFullBuckets.map(qe=>qe.splats.length),k=N.length,V=[...P.fullBuckets,...P.partiallyFullBuckets],F=M.splats.length*f,q=k*4,G=n>=1?V.length*Z.BucketStorageSizeBytes+q:0,j=F+G,ce=new ArrayBuffer(j),ye=d/(b*.5),Ie=new U;let Ne=0;for(let qe=0;qe<V.length;qe++){const ee=V[qe];Ie.fromArray(ee.center);for(let ue=0;ue<ee.splats.length;ue++){let Me=ee.splats[ue];const Ge=M.splats[Me],Ce=G+Ne*f;Z.writeSplatDataToSectionBuffer(Ge,ce,Ce,n,l,Ie,ye,d,c,u),Ne++}}if(p+=Ne,n>=1){const qe=new Uint32Array(ce,0,N.length*4);for(let ue=0;ue<N.length;ue++)qe[ue]=N[ue];const ee=new Float32Array(ce,q,V.length*Z.BucketStorageSizeFloats);for(let ue=0;ue<V.length;ue++){const Me=V[ue],Ge=ue*3;ee[Ge]=Me.center[0],ee[Ge+1]=Me.center[1],ee[Ge+2]=Me.center[2]}}h.push(ce);const de=new ArrayBuffer(Z.SectionHeaderSizeBytes);Z.writeSectionHeaderToBuffer({maxSplatCount:Ne,splatCount:Ne,bucketSize:C,bucketCount:V.length,bucketBlockSize:b,compressionScaleRange:d,storageSizeBytes:j,fullBucketCount:B,partiallyFilledBucketCount:k,sphericalHarmonicsDegree:l},n,de,0),x.push(de)}let g=0;for(let S of h)g+=S.byteLength;const m=Z.HeaderSizeBytes+Z.SectionHeaderSizeBytes*h.length+g,_=new ArrayBuffer(m);Z.writeHeaderToBuffer({versionMajor:0,versionMinor:1,maxSectionCount:h.length,sectionCount:h.length,maxSplatCount:p,splatCount:p,compressionLevel:n,sceneCenter:s,minSphericalHarmonicsCoeff:c,maxSphericalHarmonicsCoeff:u},_);let v=Z.HeaderSizeBytes;for(let S of x)new Uint8Array(_,v,Z.SectionHeaderSizeBytes).set(new Uint8Array(S)),v+=Z.SectionHeaderSizeBytes;for(let S of h)new Uint8Array(_,v,S.byteLength).set(new Uint8Array(S)),v+=S.byteLength;return new Z(_)}static computeBucketsForUncompressedSplatArray(e,t,n){let s=e.splatCount;const r=t/2,o=new U,a=new U;for(let p=0;p<s;p++){const g=e.splats[p],m=[g[De.OFFSET.X],g[De.OFFSET.Y],g[De.OFFSET.Z]];(p===0||m[0]<o.x)&&(o.x=m[0]),(p===0||m[0]>a.x)&&(a.x=m[0]),(p===0||m[1]<o.y)&&(o.y=m[1]),(p===0||m[1]>a.y)&&(a.y=m[1]),(p===0||m[2]<o.z)&&(o.z=m[2]),(p===0||m[2]>a.z)&&(a.z=m[2])}const l=new U().copy(a).sub(o),c=Math.ceil(l.y/t),u=Math.ceil(l.z/t),f=new U,d=[],h={};for(let p=0;p<s;p++){const g=e.splats[p],m=[g[De.OFFSET.X],g[De.OFFSET.Y],g[De.OFFSET.Z]],_=Math.floor((m[0]-o.x)/t),v=Math.floor((m[1]-o.y)/t),A=Math.floor((m[2]-o.z)/t);f.x=_*t+o.x+r,f.y=v*t+o.y+r,f.z=A*t+o.z+r;const S=_*(c*u)+v*u+A;let y=h[S];y||(h[S]=y={splats:[],center:f.toArray()}),y.splats.push(p),y.splats.length>=n&&(d.push(y),h[S]=null)}const x=[];for(let p in h)if(h.hasOwnProperty(p)){const g=h[p];g&&x.push(g)}return{fullBuckets:d,partiallyFullBuckets:x}}static preallocateUncompressed(e,t){const n=Z.CompressionLevels[0].SphericalHarmonicsDegrees[t],s=Z.HeaderSizeBytes+Z.SectionHeaderSizeBytes,r=s+n.BytesPerSplat*e,o=new ArrayBuffer(r);return Z.writeHeaderToBuffer({versionMajor:Z.CurrentMajorVersion,versionMinor:Z.CurrentMinorVersion,maxSectionCount:1,sectionCount:1,maxSplatCount:e,splatCount:e,compressionLevel:0,sceneCenter:new U},o),Z.writeSectionHeaderToBuffer({maxSplatCount:e,splatCount:e,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:t},0,o,Z.HeaderSizeBytes),{splatBuffer:new Z(o,!0),splatBufferDataOffsetBytes:s}}}const bm=new Uint8Array([112,108,121,10]),Mm=new Uint8Array([10,101,110,100,95,104,101,97,100,101,114,10]),Ku="end_header",ju=new Map([["char",Int8Array],["uchar",Uint8Array],["short",Int16Array],["ushort",Uint16Array],["int",Int32Array],["uint",Uint32Array],["float",Float32Array],["double",Float64Array]]),ts=(i,e)=>{const t=(1<<e)-1;return(i&t)/t},Cm=(i,e)=>{i.x=ts(e>>>21,11),i.y=ts(e>>>11,10),i.z=ts(e,11)},M1=(i,e)=>{i.x=ts(e>>>24,8),i.y=ts(e>>>16,8),i.z=ts(e>>>8,8),i.w=ts(e,8)},C1=(i,e)=>{const t=1/(Math.sqrt(2)*.5),n=(ts(e>>>20,10)-.5)*t,s=(ts(e>>>10,10)-.5)*t,r=(ts(e,10)-.5)*t,o=Math.sqrt(1-(n*n+s*s+r*r));switch(e>>>30){case 0:i.set(o,n,s,r);break;case 1:i.set(n,o,s,r);break;case 2:i.set(n,s,o,r);break;case 3:i.set(n,s,r,o);break}},vs=(i,e,t)=>i*(1-t)+e*t,Jt=(i,e)=>i.properties.find(t=>t.name===e&&t.storage)?.storage;class St{static decodeHeaderText(e){let t,n,s,r;const o=e.split(`
`).filter(f=>!f.startsWith("comment "));let a=0,l=!1;for(let f=1;f<o.length;++f){const d=o[f].split(" ");switch(d[0]){case"format":if(d[1]!=="binary_little_endian")throw new Error("Unsupported ply format");break;case"element":t={name:d[1],count:parseInt(d[2],10),properties:[],storageSizeBytes:0},t.name==="chunk"?n=t:t.name==="vertex"?s=t:t.name==="sh"&&(r=t);break;case"property":{if(!ju.has(d[1]))throw new Error(`Unrecognized property data type '${d[1]}' in ply header`);const h=ju.get(d[1]),x=h.BYTES_PER_ELEMENT*t.count;t.name==="vertex"&&(a+=h.BYTES_PER_ELEMENT),t.properties.push({type:d[1],name:d[2],storage:null,byteSize:h.BYTES_PER_ELEMENT,storageSizeByes:x}),t.storageSizeBytes+=x;break}case Ku:l=!0;break;default:throw new Error(`Unrecognized header value '${d[0]}' in ply header`)}if(l)break}let c=0,u=0;return r&&(u=r.properties.length,r.properties.length>=45?c=3:r.properties.length>=24?c=2:r.properties.length>=9&&(c=1)),{chunkElement:n,vertexElement:s,shElement:r,bytesPerSplat:a,headerSizeBytes:e.indexOf(Ku)+Ku.length+1,sphericalHarmonicsDegree:c,sphericalHarmonicsPerSplat:u}}static decodeHeader(e){const t=(h,x)=>{const p=h.length-x.length;let g,m;for(g=0;g<=p;++g){for(m=0;m<x.length&&h[g+m]===x[m];++m);if(m===x.length)return g}return-1},n=(h,x)=>{if(h.length<x.length)return!1;for(let p=0;p<x.length;++p)if(h[p]!==x[p])return!1;return!0};let s=new Uint8Array(e),r;if(s.length>=bm.length&&!n(s,bm))throw new Error("Invalid PLY header");if(r=t(s,Mm),r===-1)throw new Error("End of PLY header not found");const o=new TextDecoder("ascii").decode(s.slice(0,r)),{chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f,bytesPerSplat:d}=St.decodeHeaderText(o);return{headerSizeBytes:r+Mm.length,bytesPerSplat:d,chunkElement:a,vertexElement:l,shElement:c,sphericalHarmonicsDegree:u,sphericalHarmonicsPerSplat:f}}static readElementData(e,t,n,s,r,o=null){let a=t instanceof DataView?t:new DataView(t);s=s||0,r=r||e.count-1;for(let l=s;l<=r;++l)for(let c=0;c<e.properties.length;++c){const u=e.properties[c],f=ju.get(u.type),d=f.BYTES_PER_ELEMENT*e.count;if((!u.storage||u.storage.byteLength<d)&&(!o||o(u.name))&&(u.storage=new f(e.count)),u.storage)switch(u.type){case"char":u.storage[l]=a.getInt8(n);break;case"uchar":u.storage[l]=a.getUint8(n);break;case"short":u.storage[l]=a.getInt16(n,!0);break;case"ushort":u.storage[l]=a.getUint16(n,!0);break;case"int":u.storage[l]=a.getInt32(n,!0);break;case"uint":u.storage[l]=a.getUint32(n,!0);break;case"float":u.storage[l]=a.getFloat32(n,!0);break;case"double":u.storage[l]=a.getFloat64(n,!0);break}n+=u.byteSize}return n}static readPly(e,t=null){const n=St.decodeHeader(e);let s=St.readElementData(n.chunkElement,e,n.headerSizeBytes,null,null,t);return s=St.readElementData(n.vertexElement,e,s,null,null,t),St.readElementData(n.shElement,e,s,null,null,t),{chunkElement:n.chunkElement,vertexElement:n.vertexElement,shElement:n.shElement,sphericalHarmonicsDegree:n.sphericalHarmonicsDegree,sphericalHarmonicsPerSplat:n.sphericalHarmonicsPerSplat}}static getElementStorageArrays(e,t,n){const s={};if(t){const r=Jt(e,"min_r"),o=Jt(e,"min_g"),a=Jt(e,"min_b"),l=Jt(e,"max_r"),c=Jt(e,"max_g"),u=Jt(e,"max_b"),f=Jt(e,"min_x"),d=Jt(e,"min_y"),h=Jt(e,"min_z"),x=Jt(e,"max_x"),p=Jt(e,"max_y"),g=Jt(e,"max_z"),m=Jt(e,"min_scale_x"),_=Jt(e,"min_scale_y"),v=Jt(e,"min_scale_z"),A=Jt(e,"max_scale_x"),S=Jt(e,"max_scale_y"),y=Jt(e,"max_scale_z"),M=Jt(t,"packed_position"),E=Jt(t,"packed_rotation"),b=Jt(t,"packed_scale"),C=Jt(t,"packed_color");s.colorExtremes={minR:r,maxR:l,minG:o,maxG:c,minB:a,maxB:u},s.positionExtremes={minX:f,maxX:x,minY:d,maxY:p,minZ:h,maxZ:g},s.scaleExtremes={minScaleX:m,maxScaleX:A,minScaleY:_,maxScaleY:S,minScaleZ:v,maxScaleZ:y},s.position=M,s.rotation=E,s.scale=b,s.color=C}if(n){const r={};for(let o=0;o<45;o++){const a=`f_rest_${o}`,l=Jt(n,a);if(l)r[a]=l;else break}s.sh=r}return s}static decompressBaseSplat=(function(){const e=new U,t=new Gt,n=new U,s=new Zt,r=De.OFFSET;return function(o,a,l,c,u,f,d,h,x,p){p=p||De.createSplat();const g=Math.floor((a+o)/256);return Cm(e,l[o]),C1(t,d[o]),Cm(n,u[o]),M1(s,x[o]),p[r.X]=vs(c.minX[g],c.maxX[g],e.x),p[r.Y]=vs(c.minY[g],c.maxY[g],e.y),p[r.Z]=vs(c.minZ[g],c.maxZ[g],e.z),p[r.ROTATION0]=t.x,p[r.ROTATION1]=t.y,p[r.ROTATION2]=t.z,p[r.ROTATION3]=t.w,p[r.SCALE0]=Math.exp(vs(f.minScaleX[g],f.maxScaleX[g],n.x)),p[r.SCALE1]=Math.exp(vs(f.minScaleY[g],f.maxScaleY[g],n.y)),p[r.SCALE2]=Math.exp(vs(f.minScaleZ[g],f.maxScaleZ[g],n.z)),h.minR&&h.maxR?p[r.FDC0]=jt(Math.round(vs(h.minR[g],h.maxR[g],s.x)*255),0,255):p[r.FDC0]=jt(Math.floor(s.x*255),0,255),h.minG&&h.maxG?p[r.FDC1]=jt(Math.round(vs(h.minG[g],h.maxG[g],s.y)*255),0,255):p[r.FDC1]=jt(Math.floor(s.y*255),0,255),h.minB&&h.maxB?p[r.FDC2]=jt(Math.round(vs(h.minB[g],h.maxB[g],s.z)*255),0,255):p[r.FDC2]=jt(Math.floor(s.z*255),0,255),p[r.OPACITY]=jt(Math.floor(s.w*255),0,255),p}})();static decompressSphericalHarmonics=(function(){const e=[0,3,8,15],t=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(n,s,r,o,a){a=a||De.createSplat();let l=e[r],c=e[o];for(let u=0;u<3;++u)for(let f=0;f<15;++f){const d=t[u*15+f];f<l&&f<c&&(a[De.OFFSET.FRC0+d]=s[u*c+f][n]*(8/255)-4)}return a}})();static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l,c=null){St.readElementData(t,o,0,n,s,c);const u=Z.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,{positionExtremes:f,scaleExtremes:d,colorExtremes:h,position:x,rotation:p,scale:g,color:m}=St.getElementStorageArrays(e,t),_=De.createSplat();for(let v=n;v<=s;++v){St.decompressBaseSplat(v,r,x,f,g,d,p,h,m,_);const A=v*u+l;Z.writeSplatDataToSectionBuffer(_,a,A,0,0)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a,l=null){St.readElementData(t,o,0,n,s,l);const{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:d,rotation:h,scale:x,color:p}=St.getElementStorageArrays(e,t);for(let g=n;g<=s;++g){const m=De.createSplat();St.decompressBaseSplat(g,r,d,c,x,u,h,f,p,m),a.addSplat(m)}}static parseSphericalHarmonicsToUncompressedSplatArraySection(e,t,n,s,r,o,a,l,c,u=null){St.readElementData(t,r,o,n,s,u);const{sh:f}=St.getElementStorageArrays(e,void 0,t),d=Object.values(f);for(let h=n;h<=s;++h)St.decompressSphericalHarmonics(h,d,a,l,c.splats[h])}static parseToUncompressedSplatArray(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=St.readPly(e);t=Math.min(t,o);const a=new De(t),{positionExtremes:l,scaleExtremes:c,colorExtremes:u,position:f,rotation:d,scale:h,color:x}=St.getElementStorageArrays(n,s);let p;if(t>0){const{sh:g}=St.getElementStorageArrays(n,void 0,r);p=Object.values(g)}for(let g=0;g<s.count;++g){a.addDefaultSplat();const m=a.getSplat(a.splatCount-1);St.decompressBaseSplat(g,0,f,l,h,c,d,u,x,m),t>0&&St.decompressSphericalHarmonics(g,p,t,o,m)}return a}static parseToUncompressedSplatBuffer(e,t){const{chunkElement:n,vertexElement:s,shElement:r,sphericalHarmonicsDegree:o}=St.readPly(e);t=Math.min(t,o);const{splatBuffer:a,splatBufferDataOffsetBytes:l}=Z.preallocateUncompressed(s.count,t),{positionExtremes:c,scaleExtremes:u,colorExtremes:f,position:d,rotation:h,scale:x,color:p}=St.getElementStorageArrays(n,s);let g;if(t>0){const{sh:v}=St.getElementStorageArrays(n,void 0,r);g=Object.values(v)}const m=Z.CompressionLevels[0].SphericalHarmonicsDegrees[t].BytesPerSplat,_=De.createSplat(t);for(let v=0;v<s.count;++v){St.decompressBaseSplat(v,0,d,c,x,u,h,f,p,_),t>0&&St.decompressSphericalHarmonics(v,g,t,o,_);const A=v*m+l;Z.writeSplatDataToSectionBuffer(_,a.bufferData,A,0,t)}return a}}const Wn={INRIAV1:0,INRIAV2:1,PlayCanvasCompressed:2},[rx,uh,fh,dh,hh,ph,mh]=[0,1,2,3,4,5,6],Tm={double:rx,int:uh,uint:fh,float:dh,short:hh,ushort:ph,uchar:mh},T1={[rx]:8,[uh]:4,[fh]:4,[dh]:4,[hh]:2,[ph]:2,[mh]:1};class Mt{static HeaderEndToken="end_header";static decodeSectionHeader(e,t,n=0){const s=[];let r=!1,o=-1,a=0,l=!1,c=null;const u=[],f=[],d=[],h={};for(let m=n;m<e.length;m++){const _=e[m].trim();if(_.startsWith("element"))if(r){o--;break}else{r=!0,n=m,o=m;const v=_.split(" ");let A=0;for(let S of v){const y=S.trim();y.length>0&&(A++,A===2?c=y:A===3&&(a=parseInt(y)))}}else if(_.startsWith("property")){const v=_.match(/(\w+)\s+(\w+)\s+(\w+)/);if(v){const A=v[2],S=v[3];d.push(S);const y=t[S];h[S]=A;const M=Tm[A];y!==void 0&&(u.push(y),f[y]=M)}}if(_===Mt.HeaderEndToken){l=!0;break}r&&(s.push(_),o++)}const x=[];let p=0;for(let m of d){const _=h[m];if(h.hasOwnProperty(m)){const v=t[m];v!==void 0&&(x[v]=p)}p+=T1[Tm[_]]}const g=Mt.decodeSphericalHarmonicsFromSectionHeader(d,t);return{headerLines:s,headerStartLine:n,headerEndLine:o,fieldTypes:f,fieldIds:u,fieldOffsets:x,bytesPerVertex:p,vertexCount:a,dataSizeBytes:p*a,endOfHeader:l,sectionName:c,sphericalHarmonicsDegree:g.degree,sphericalHarmonicsCoefficientsPerChannel:g.coefficientsPerChannel,sphericalHarmonicsDegree1Fields:g.degree1Fields,sphericalHarmonicsDegree2Fields:g.degree2Fields}}static decodeSphericalHarmonicsFromSectionHeader(e,t){let n=0,s=0;for(let l of e)l.startsWith("f_rest")&&n++;s=n/3;let r=0;s>=3&&(r=1),s>=8&&(r=2);let o=[],a=[];for(let l=0;l<3;l++){if(r>=1)for(let c=0;c<3;c++)o.push(t["f_rest_"+(c+s*l)]);if(r>=2)for(let c=0;c<5;c++)a.push(t["f_rest_"+(c+s*l+3)])}return{degree:r,coefficientsPerChannel:s,degree1Fields:o,degree2Fields:a}}static getHeaderSectionNames(e){const t=[];for(let n of e)if(n.startsWith("element")){const s=n.split(" ");let r=0;for(let o of s){const a=o.trim();a.length>0&&(r++,r===2&&t.push(a))}}return t}static checkTextForEndHeader(e){return!!e.includes(Mt.HeaderEndToken)}static checkBufferForEndHeader(e,t,n,s){const r=new Uint8Array(e,Math.max(0,t-n),n),o=s.decode(r);return Mt.checkTextForEndHeader(o)}static extractHeaderFromBufferToText(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,Mt.checkBufferForEndHeader(e,n,r*2,t))break}return s}static readHeaderFromBuffer(e){const t=new TextDecoder;let n=0,s="";const r=100;for(;;){if(n+r>=e.byteLength)throw new Error("End of file reached while searching for end of header");const o=new Uint8Array(e,n,r);if(s+=t.decode(o),n+=r,Mt.checkBufferForEndHeader(e,n,r*2,t))break}return s}static convertHeaderTextToLines(e){const t=e.split(`
`),n=[];for(let s=0;s<t.length;s++){const r=t[s].trim();if(n.push(r),r===Mt.HeaderEndToken)break}return n}static determineHeaderFormatFromHeaderText(e){const t=Mt.convertHeaderTextToLines(e);let n=Wn.INRIAV1;for(let s=0;s<t.length;s++){const r=t[s].trim();if(r.startsWith("element chunk")||r.match(/[A-Za-z]*packed_[A-Za-z]*/))n=Wn.PlayCanvasCompressed;else if(r.startsWith("element codebook_centers"))n=Wn.INRIAV2;else if(r===Mt.HeaderEndToken)break}return n}static determineHeaderFormatFromPlyBuffer(e){const t=Mt.extractHeaderFromBufferToText(e);return Mt.determineHeaderFormatFromHeaderText(t)}static readVertex(e,t,n,s,r,o,a=!0){const l=n*t.bytesPerVertex+s,c=t.fieldOffsets,u=t.fieldTypes;for(let f of r){const d=u[f];d===dh?o[f]=e.getFloat32(l+c[f],!0):d===hh?o[f]=e.getInt16(l+c[f],!0):d===ph?o[f]=e.getUint16(l+c[f],!0):d===uh?o[f]=e.getInt32(l+c[f],!0):d===fh?o[f]=e.getUint32(l+c[f],!0):d===mh&&(a?o[f]=e.getUint8(l+c[f])/255:o[f]=e.getUint8(l+c[f]))}}}const ox=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0"],E1=ox.map((i,e)=>e),[Em,w1,R1,I1,D1,P1,F1,L1,B1,U1,wm,O1,N1,Rm,Im,z1,k1,H1]=E1;class En{static decodeHeaderLines(e){let t=0;e.forEach(u=>{u.includes("f_rest_")&&t++});let n=0;t>=45?n=45:t>=24?n=24:t>=9&&(n=9);let r=Array.from(Array(Math.max(n-1,0))).map((u,f)=>`f_rest_${f+1}`);const o=[...ox,...r],a=o.map((u,f)=>f),l=a.reduce((u,f)=>(u[o[f]]=f,u),{}),c=Mt.decodeSectionHeader(e,l,0);return c.splatCount=c.vertexCount,c.bytesPerSplat=c.bytesPerVertex,c.fieldsToReadIndexes=a,c}static decodeHeaderText(e){const t=Mt.convertHeaderTextToLines(e),n=En.decodeHeaderLines(t);return n.headerText=e,n.headerSizeBytes=e.indexOf(Mt.HeaderEndToken)+Mt.HeaderEndToken.length+1,n}static decodeHeaderFromBuffer(e){const t=Mt.readHeaderFromBuffer(e);return En.decodeHeaderText(t)}static findSplatData(e,t){return new DataView(e,t.headerSizeBytes)}static parseToUncompressedSplatBufferSection(e,t,n,s,r,o,a,l=0){l=Math.min(l,e.sphericalHarmonicsDegree);const c=Z.CompressionLevels[0].SphericalHarmonicsDegrees[l].BytesPerSplat;for(let u=t;u<=n;u++){const f=En.parseToUncompressedSplat(s,u,e,r,l),d=u*c+a;Z.writeSplatDataToSectionBuffer(f,o,d,0,l)}}static parseToUncompressedSplatArraySection(e,t,n,s,r,o,a=0){a=Math.min(a,e.sphericalHarmonicsDegree);for(let l=t;l<=n;l++){const c=En.parseToUncompressedSplat(s,l,e,r,a);o.addSplat(c)}}static decodeSectionSplatData(e,t,n,s,r=!0){if(s=Math.min(s,n.sphericalHarmonicsDegree),r){const o=new De(s);for(let a=0;a<t;a++){const l=En.parseToUncompressedSplat(e,a,n,0,s);o.addSplat(l)}return o}else{const{splatBuffer:o,splatBufferDataOffsetBytes:a}=Z.preallocateUncompressed(t,s);return En.parseToUncompressedSplatBufferSection(n,0,t-1,e,0,o.bufferData,a,s),o}}static parseToUncompressedSplat=(function(){let e=[];const t=new Gt,n=De.OFFSET.X,s=De.OFFSET.Y,r=De.OFFSET.Z,o=De.OFFSET.SCALE0,a=De.OFFSET.SCALE1,l=De.OFFSET.SCALE2,c=De.OFFSET.ROTATION0,u=De.OFFSET.ROTATION1,f=De.OFFSET.ROTATION2,d=De.OFFSET.ROTATION3,h=De.OFFSET.FDC0,x=De.OFFSET.FDC1,p=De.OFFSET.FDC2,g=De.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=De.OFFSET.FRC0+_;return function(_,v,A,S=0,y=0){y=Math.min(y,A.sphericalHarmonicsDegree),En.readSplat(_,A,v,S,e);const M=De.createSplat(y);if(e[Em]!==void 0?(M[o]=Math.exp(e[Em]),M[a]=Math.exp(e[w1]),M[l]=Math.exp(e[R1])):(M[o]=.01,M[a]=.01,M[l]=.01),e[wm]!==void 0){const E=.28209479177387814;M[h]=(.5+E*e[wm])*255,M[x]=(.5+E*e[O1])*255,M[p]=(.5+E*e[N1])*255}else e[Im]!==void 0?(M[h]=e[Im]*255,M[x]=e[z1]*255,M[p]=e[k1]*255):(M[h]=0,M[x]=0,M[p]=0);if(e[Rm]!==void 0&&(M[g]=1/(1+Math.exp(-e[Rm]))*255),M[h]=jt(Math.floor(M[h]),0,255),M[x]=jt(Math.floor(M[x]),0,255),M[p]=jt(Math.floor(M[p]),0,255),M[g]=jt(Math.floor(M[g]),0,255),y>=1&&e[H1]!==void 0){for(let E=0;E<9;E++)M[m[E]]=e[A.sphericalHarmonicsDegree1Fields[E]];if(y>=2)for(let E=0;E<15;E++)M[m[9+E]]=e[A.sphericalHarmonicsDegree2Fields[E]]}return t.set(e[I1],e[D1],e[P1],e[F1]),t.normalize(),M[c]=t.x,M[u]=t.y,M[f]=t.z,M[d]=t.w,M[n]=e[L1],M[s]=e[B1],M[r]=e[U1],M}})();static readSplat(e,t,n,s,r){return Mt.readVertex(e,t,n,s,t.fieldsToReadIndexes,r,!0)}static parseToUncompressedSplatArray(e,t=0){const{header:n,splatCount:s,splatData:r}=Dm(e);return En.decodeSectionSplatData(r,s,n,t,!0)}static parseToUncompressedSplatBuffer(e,t=0){const{header:n,splatCount:s,splatData:r}=Dm(e);return En.decodeSectionSplatData(r,s,n,t,!1)}}function Dm(i){const e=En.decodeHeaderFromBuffer(i),t=e.splatCount,n=En.findSplatData(i,e);return{header:e,splatCount:t,splatData:n}}const ax=["features_dc","features_rest_0","features_rest_1","features_rest_2","features_rest_3","features_rest_4","features_rest_5","features_rest_6","features_rest_7","features_rest_8","features_rest_9","features_rest_10","features_rest_11","features_rest_12","features_rest_13","features_rest_14","opacity","scaling","rotation_re","rotation_im"],Xl=ax.map((i,e)=>e),[ql,V1,G1,Pm,Yl,W1,$u]=[0,1,4,16,17,18,19],lx=["scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","red","green","blue","f_rest_0","f_rest_1","f_rest_2","f_rest_3","f_rest_4","f_rest_5","f_rest_6","f_rest_7","f_rest_8","f_rest_9","f_rest_10","f_rest_11","f_rest_12","f_rest_13","f_rest_14","f_rest_15","f_rest_16","f_rest_17","f_rest_18","f_rest_19","f_rest_20","f_rest_21","f_rest_22","f_rest_23","f_rest_24","f_rest_25","f_rest_26","f_rest_27","f_rest_28","f_rest_29","f_rest_30","f_rest_31","f_rest_32","f_rest_33","f_rest_34","f_rest_35","f_rest_36","f_rest_37","f_rest_38","f_rest_39","f_rest_40","f_rest_41","f_rest_42","f_rest_43","f_rest_44","f_rest_45"],dd=lx.map((i,e)=>e),[Fm,X1,q1,Y1,Q1,K1,j1,$1,Z1,J1,hd,cx,ux,Lm]=dd,Bm=hd,eE=cx,tE=ux,Ql=i=>{const e=(31744&i)>>10,t=1023&i;return(i>>15?-1:1)*(e?e===31?t?NaN:1/0:Math.pow(2,e-15)*(1+t/1024):t/1024*6103515625e-14)};class ui{static decodeSectionHeadersFromHeaderLines(e){const t=dd.reduce((u,f)=>(u[lx[f]]=f,u),{}),n=Xl.reduce((u,f)=>(u[ax[f]]=f,u),{}),s=Mt.getHeaderSectionNames(e);let r;for(let u=0;u<s.length;u++)s[u]==="codebook_centers"&&(r=u);let o=0,a=!1;const l=[];let c=0;for(;!a;){let u;c===r?u=Mt.decodeSectionHeader(e,n,o):u=Mt.decodeSectionHeader(e,t,o),a=u.endOfHeader,o=u.headerEndLine+1,a||(u.splatCount=u.vertexCount,u.bytesPerSplat=u.bytesPerVertex),l.push(u),c++}return l}static decodeSectionHeadersFromHeaderText(e){const t=Mt.convertHeaderTextToLines(e);return ui.decodeSectionHeadersFromHeaderLines(t)}static getSplatCountFromSectionHeaders(e){let t=0;for(let n of e)n.sectionName!=="codebook_centers"&&(t+=n.vertexCount);return t}static decodeHeaderFromHeaderText(e){const t=e.indexOf(Mt.HeaderEndToken)+Mt.HeaderEndToken.length+1,n=ui.decodeSectionHeadersFromHeaderText(e),s=ui.getSplatCountFromSectionHeaders(n);return{headerSizeBytes:t,sectionHeaders:n,splatCount:s}}static decodeHeaderFromBuffer(e){const t=Mt.readHeaderFromBuffer(e);return ui.decodeHeaderFromHeaderText(t)}static findVertexData(e,t,n){let s=t.headerSizeBytes;for(let r=0;r<n&&r<t.sectionHeaders.length;r++){const o=t.sectionHeaders[r];s+=o.dataSizeBytes}return new DataView(e,s,t.sectionHeaders[n].dataSizeBytes)}static decodeCodeBook(e,t){const n=[],s=[];for(let r=0;r<t.vertexCount;r++){Mt.readVertex(e,t,r,0,Xl,n);for(let o of Xl){const a=Xl[o];let l=s[a];l||(s[a]=l=[]),l.push(n[o])}}for(let r=0;r<s.length;r++){const o=s[r],a=.28209479177387814;for(let l=0;l<o.length;l++){const c=Ql(o[l]);r===Pm?o[l]=Math.round(1/(1+Math.exp(-c))*255):r===ql?o[l]=Math.round((.5+a*c)*255):r===Yl?o[l]=Math.exp(c):o[l]=c}}return s}static decodeSectionSplatData(e,t,n,s,r){r=Math.min(r,n.sphericalHarmonicsDegree);const o=new De(r);for(let a=0;a<t;a++){const l=ui.parseToUncompressedSplat(e,a,n,s,0,r);o.addSplat(l)}return o}static parseToUncompressedSplat=(function(){let e=[];const t=new Gt,n=De.OFFSET.X,s=De.OFFSET.Y,r=De.OFFSET.Z,o=De.OFFSET.SCALE0,a=De.OFFSET.SCALE1,l=De.OFFSET.SCALE2,c=De.OFFSET.ROTATION0,u=De.OFFSET.ROTATION1,f=De.OFFSET.ROTATION2,d=De.OFFSET.ROTATION3,h=De.OFFSET.FDC0,x=De.OFFSET.FDC1,p=De.OFFSET.FDC2,g=De.OFFSET.OPACITY,m=[];for(let _=0;_<45;_++)m[_]=De.OFFSET.FRC0+_;return function(_,v,A,S,y=0,M=0){M=Math.min(M,A.sphericalHarmonicsDegree),ui.readSplat(_,A,v,y,e);const E=De.createSplat(M);if(e[Fm]!==void 0?(E[o]=S[Yl][e[Fm]],E[a]=S[Yl][e[X1]],E[l]=S[Yl][e[q1]]):(E[o]=.01,E[a]=.01,E[l]=.01),e[hd]!==void 0?(E[h]=S[ql][e[hd]],E[x]=S[ql][e[cx]],E[p]=S[ql][e[ux]]):e[Bm]!==void 0?(E[h]=e[Bm]*255,E[x]=e[eE]*255,E[p]=e[tE]*255):(E[h]=0,E[x]=0,E[p]=0),e[Lm]!==void 0&&(E[g]=S[Pm][e[Lm]]),E[h]=jt(Math.floor(E[h]),0,255),E[x]=jt(Math.floor(E[x]),0,255),E[p]=jt(Math.floor(E[p]),0,255),E[g]=jt(Math.floor(E[g]),0,255),M>=1&&A.sphericalHarmonicsDegree>=1){for(let N=0;N<9;N++){const k=S[V1+N%3];E[m[N]]=k[e[A.sphericalHarmonicsDegree1Fields[N]]]}if(M>=2&&A.sphericalHarmonicsDegree>=2)for(let N=0;N<15;N++){const k=S[G1+N%5];E[m[9+N]]=k[e[A.sphericalHarmonicsDegree2Fields[N]]]}}const b=S[W1][e[Y1]],C=S[$u][e[Q1]],P=S[$u][e[K1]],B=S[$u][e[j1]];return t.set(b,C,P,B),t.normalize(),E[c]=t.x,E[u]=t.y,E[f]=t.z,E[d]=t.w,E[n]=Ql(e[$1]),E[s]=Ql(e[Z1]),E[r]=Ql(e[J1]),E}})();static readSplat(e,t,n,s,r){return Mt.readVertex(e,t,n,s,dd,r,!1)}static parseToUncompressedSplatArray(e,t=0){const n=[],s=ui.decodeHeaderFromBuffer(e,t);let r;for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName==="codebook_centers"){const c=ui.findVertexData(e,s,a);r=ui.decodeCodeBook(c,l)}}for(let a=0;a<s.sectionHeaders.length;a++){const l=s.sectionHeaders[a];if(l.sectionName!=="codebook_centers"){const c=l.vertexCount,u=ui.findVertexData(e,s,a),f=ui.decodeSectionSplatData(u,c,l,r,t);n.push(f)}}const o=new De(t);for(let a of n)for(let l of a.splats)o.addSplat(l);return o}}class Um{static parseToUncompressedSplatArray(e,t=0){const n=Mt.determineHeaderFormatFromPlyBuffer(e);if(n===Wn.PlayCanvasCompressed)return St.parseToUncompressedSplatArray(e,t);if(n===Wn.INRIAV1)return En.parseToUncompressedSplatArray(e,t);if(n===Wn.INRIAV2)return ui.parseToUncompressedSplatArray(e,t)}static parseToUncompressedSplatBuffer(e,t=0){const n=Mt.determineHeaderFormatFromPlyBuffer(e);if(n===Wn.PlayCanvasCompressed)return St.parseToUncompressedSplatBuffer(e,t);if(n===Wn.INRIAV1)return En.parseToUncompressedSplatBuffer(e,t);if(n===Wn.INRIAV2)throw new Error("parseToUncompressedSplatBuffer() is not implemented for INRIA V2 PLY files")}}class gh{constructor(e,t,n,s){this.sectionCount=e,this.sectionFilters=t,this.groupingParameters=n,this.partitionGenerator=s}partitionUncompressedSplatArray(e){let t,n,s;if(this.partitionGenerator){const o=this.partitionGenerator(e);t=o.groupingParameters,n=o.sectionCount,s=o.sectionFilters}else t=this.groupingParameters,n=this.sectionCount,s=this.sectionFilters;const r=[];for(let o=0;o<n;o++){const a=new De(e.sphericalHarmonicsDegree),l=s[o];for(let c=0;c<e.splatCount;c++)l(c)&&a.addSplat(e.splats[c]);r.push(a)}return{splatArrays:r,parameters:t}}static getStandardPartitioner(e=0,t=new U,n=Z.BucketBlockSize,s=Z.BucketSize){const r=o=>{const a=De.OFFSET.X,l=De.OFFSET.Y,c=De.OFFSET.Z;e<=0&&(e=o.splatCount);const u=new U,f=.5,d=m=>{m.x=Math.floor(m.x/f)*f,m.y=Math.floor(m.y/f)*f,m.z=Math.floor(m.z/f)*f};o.splats.forEach(m=>{u.set(m[a],m[l],m[c]).sub(t),d(u),m.centerDist=u.lengthSq()}),o.splats.sort((m,_)=>{let v=m.centerDist,A=_.centerDist;return v>A?1:-1});const h=[],x=[];e=Math.min(o.splatCount,e);const p=Math.ceil(o.splatCount/e);let g=0;for(let m=0;m<p;m++){let _=g;h.push(v=>v>=_&&v<_+e),x.push({blocksSize:n,bucketSize:s}),g+=e}return{sectionCount:h.length,sectionFilters:h,groupingParameters:x}};return new gh(void 0,void 0,void 0,r)}}class dl{constructor(e,t,n,s,r,o,a){this.splatPartitioner=e,this.alphaRemovalThreshold=t,this.compressionLevel=n,this.sectionSize=s,this.sceneCenter=r?new U().copy(r):void 0,this.blockSize=o,this.bucketSize=a}generateFromUncompressedSplatArray(e){const t=this.splatPartitioner.partitionUncompressedSplatArray(e);return Z.generateFromUncompressedSplatArrays(t.splatArrays,this.alphaRemovalThreshold,this.compressionLevel,this.sceneCenter,this.blockSize,this.bucketSize,t.parameters)}static getStandardGenerator(e=1,t=1,n=0,s=new U,r=Z.BucketBlockSize,o=Z.BucketSize){const a=gh.getStandardPartitioner(n,s,r,o);return new dl(a,e,t,n,s,r,o)}}const an={Downloading:0,Processing:1,Done:2};class bc extends Error{constructor(e){super(e)}}const qt={ProgressiveToSplatBuffer:0,ProgressiveToSplatArray:1,DownloadBeforeProcessing:2};function Om(i,e){let t=0;for(let s of i)t+=s.sizeBytes;(!e||e.byteLength<t)&&(e=new ArrayBuffer(t));let n=0;for(let s of i)new Uint8Array(e,n,s.sizeBytes).set(s.data),n+=s.sizeBytes;return e}function Nm(i,e,t,n,s,r,o,a){return e?dl.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):Z.generateFromUncompressedSplatArrays([i],t,0,new U)}class xh{static loadFromURL(e,t,n,s,r,o,a=!0,l=0,c,u,f,d,h){let x;!n&&!a?x=qt.DownloadBeforeProcessing:a?x=qt.ProgressiveToSplatArray:x=qt.ProgressiveToSplatBuffer;const p=Ot.ProgressiveLoadSectionSize,g=Z.HeaderSizeBytes+Z.SectionHeaderSizeBytes,m=1;let _,v,A,S,y,M=0,E=0,b=0,C=!1,P=!1,B=!1;const N=oh();let k=0,V=0,F=0,q=0,G="",j=null,ce=[],ye;const Ie=new TextDecoder,Ne=(de,qe,ee)=>{const ue=de>=100;if(ee&&(ce.push({data:ee,sizeBytes:ee.byteLength,startBytes:F,endBytes:F+ee.byteLength}),F+=ee.byteLength),x===qt.DownloadBeforeProcessing)ue&&N.resolve(ce);else{if(C){if(_===Wn.PlayCanvasCompressed&&!P){const Me=j.headerSizeBytes+j.chunkElement.storageSizeBytes;y=Om(ce,y),y.byteLength>=Me&&(St.readElementData(j.chunkElement,y,j.headerSizeBytes),k=Me,V=Me,P=!0)}}else if(G+=Ie.decode(ee),Mt.checkTextForEndHeader(G)){if(_=Mt.determineHeaderFormatFromHeaderText(G),_===Wn.INRIAV1)j=En.decodeHeaderText(G),l=Math.min(l,j.sphericalHarmonicsDegree),M=j.splatCount,P=!0,q=j.headerSizeBytes+j.bytesPerSplat*M;else if(_===Wn.PlayCanvasCompressed){if(j=St.decodeHeaderText(G),l=Math.min(l,j.sphericalHarmonicsDegree),x===qt.ProgressiveToSplatBuffer&&l>0)throw new bc("PlyLoader.loadFromURL() -> Selected PLY format has spherical harmonics data that cannot be progressively loaded.");M=j.vertexElement.count,q=j.headerSizeBytes+j.bytesPerSplat*M+j.chunkElement.storageSizeBytes}else{if(x===qt.ProgressiveToSplatBuffer)throw new bc("PlyLoader.loadFromURL() -> Selected PLY format cannot be progressively loaded.");x=qt.DownloadBeforeProcessing;return}if(x===qt.ProgressiveToSplatBuffer){const Me=Z.CompressionLevels[0].SphericalHarmonicsDegrees[l],Ge=g+Me.BytesPerSplat*M;A=new ArrayBuffer(Ge),Z.writeHeaderToBuffer({versionMajor:Z.CurrentMajorVersion,versionMinor:Z.CurrentMinorVersion,maxSectionCount:m,sectionCount:m,maxSplatCount:M,splatCount:0,compressionLevel:0,sceneCenter:new U},A)}else ye=new De(l);k=j.headerSizeBytes,V=j.headerSizeBytes,C=!0}if(C&&P&&ce.length>0&&(v=Om(ce,v),F-k>p||F>=q&&!B||ue)){const Ge=B?j.sphericalHarmonicsPerSplat:j.bytesPerSplat,Ye=(B?F:Math.min(q,F))-V,O=Math.floor(Ye/Ge),z=O*Ge,Q=F-V-z,w=V-ce[0].startBytes,ne=new DataView(v,w,z);if(B)_===Wn.PlayCanvasCompressed&&x===qt.ProgressiveToSplatArray&&(St.parseSphericalHarmonicsToUncompressedSplatArraySection(j.chunkElement,j.shElement,b,b+O-1,ne,0,l,j.sphericalHarmonicsDegree,ye),b+=O);else{if(x===qt.ProgressiveToSplatBuffer){const oe=Z.CompressionLevels[0].SphericalHarmonicsDegrees[l],he=E*oe.BytesPerSplat+g;_===Wn.PlayCanvasCompressed?St.parseToUncompressedSplatBufferSection(j.chunkElement,j.vertexElement,0,O-1,E,ne,A,he):En.parseToUncompressedSplatBufferSection(j,0,O-1,ne,0,A,he,l)}else _===Wn.PlayCanvasCompressed?St.parseToUncompressedSplatArraySection(j.chunkElement,j.vertexElement,0,O-1,E,ne,ye):En.parseToUncompressedSplatArraySection(j,0,O-1,ne,0,ye,l);E+=O,x===qt.ProgressiveToSplatBuffer&&(S||(Z.writeSectionHeaderToBuffer({maxSplatCount:M,splatCount:E,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0,sphericalHarmonicsDegree:l},0,A,Z.HeaderSizeBytes),S=new Z(A,!1)),S.updateLoadedCounts(1,E)),F>=q&&(B=!0)}if(Q===0)ce=[];else{let oe=[],he=0;for(let ie=ce.length-1;ie>=0;ie--){const pe=ce[ie];if(he+=pe.sizeBytes,oe.unshift(pe),he>=Q)break}ce=oe}k+=p,V+=z}s&&S&&s(S,ue),ue&&(x===qt.ProgressiveToSplatBuffer?N.resolve(S):N.resolve(ye))}t&&t(de,qe,an.Downloading)};return t&&t(0,"0%",an.Downloading),Qc(e,Ne,!1,c).then(()=>(t&&t(0,"0%",an.Processing),N.promise.then(de=>{if(t&&t(100,"100%",an.Done),x===qt.DownloadBeforeProcessing){const qe=ce.map(ee=>ee.data);return new Blob(qe).arrayBuffer().then(ee=>xh.loadFromFileData(ee,r,o,a,l,u,f,d,h))}else return x===qt.ProgressiveToSplatBuffer?de:pi(()=>Nm(de,a,r,o,u,f,d,h))})))}static loadFromFileData(e,t,n,s,r=0,o,a,l,c){return s?pi(()=>Um.parseToUncompressedSplatArray(e,r)).then(u=>Nm(u,s,t,n,o,a,l,c)):pi(()=>Um.parseToUncompressedSplatBuffer(e,r))}}const nE=i=>new ReadableStream({async start(e){e.enqueue(i),e.close()}});async function iE(i){try{const e=nE(i);if(!e)throw new Error("Failed to create stream from data");return await sE(e)}catch(e){throw console.error("Error decompressing gzipped data:",e),e}}async function sE(i){const e=i.pipeThrough(new DecompressionStream("gzip")),n=await new Response(e).arrayBuffer();return new Uint8Array(n)}const rE=1347635022,oE=1,aE=.15;function lE(i){const e=i>>15&1,t=i>>10&31,n=i&1023,s=e===1?-1:1;return t===0?s*Math.pow(2,-14)*n/1024:t===31?n!==0?NaN:s*(1/0):s*Math.pow(2,t-15)*(1+n/1024)}function cE(i){return(i-128)/128}function zr(i){switch(i){case 0:return 0;case 1:return 3;case 2:return 8;case 3:return 15;default:return console.error(`[SPZ: ERROR] Unsupported SH degree: ${i}`),0}}const uE=(function(){let i=[];const e=new Gt,t=De.OFFSET.X,n=De.OFFSET.Y,s=De.OFFSET.Z,r=De.OFFSET.SCALE0,o=De.OFFSET.SCALE1,a=De.OFFSET.SCALE2,l=De.OFFSET.ROTATION0,c=De.OFFSET.ROTATION1,u=De.OFFSET.ROTATION2,f=De.OFFSET.ROTATION3,d=De.OFFSET.FDC0,h=De.OFFSET.FDC1,x=De.OFFSET.FDC2,p=De.OFFSET.OPACITY,g=[zr(0),zr(1),zr(2),zr(3)],m=[0,1,2,9,10,11,12,13,24,25,26,27,28,29,30,3,4,5,14,15,16,17,18,31,32,33,34,35,36,37,6,7,8,19,20,21,22,23,38,39,40,41,42,43,44];return function(_,v,A){A=Math.min(v,A);const S=De.createSplat(A);_.scale[0]!==void 0?(S[r]=_.scale[0],S[o]=_.scale[1],S[a]=_.scale[2]):(S[r]=.01,S[o]=.01,S[a]=.01),_.color[0]!==void 0?(S[d]=_.color[0],S[h]=_.color[1],S[x]=_.color[2]):i[RED]!==void 0?(S[d]=i[RED]*255,S[h]=i[GREEN]*255,S[x]=i[BLUE]*255):(S[d]=0,S[h]=0,S[x]=0),_.alpha!==void 0&&(S[p]=_.alpha),S[d]=jt(Math.floor(S[d]),0,255),S[h]=jt(Math.floor(S[h]),0,255),S[x]=jt(Math.floor(S[x]),0,255),S[p]=jt(Math.floor(S[p]),0,255);let y=g[A],M=g[v];for(let E=0;E<3;++E)for(let b=0;b<15;++b){const C=m[E*15+b];b<y&&b<M&&(S[De.OFFSET.FRC0+C]=_.sh[E*M+b])}return e.set(_.rotation[3],_.rotation[0],_.rotation[1],_.rotation[2]),e.normalize(),S[l]=e.x,S[c]=e.y,S[u]=e.z,S[f]=e.w,S[t]=_.position[0],S[n]=_.position[1],S[s]=_.position[2],S}})();function fE(i,e,t,n){return!(i.positions.length!==e*3*(n?2:3)||i.scales.length!==e*3||i.rotations.length!==e*3||i.alphas.length!==e||i.colors.length!==e*3||i.sh.length!==e*t*3)}function zm(i,e,t,n,s){e=Math.min(e,i.shDegree);const r=i.numPoints,o=zr(i.shDegree),a=i.positions.length===r*3*2;if(!fE(i,r,o,a))return null;const l={position:[],scale:[],rotation:[],alpha:void 0,color:[],sh:[]};let c;a&&(c=new Uint16Array(i.positions.buffer,i.positions.byteOffset,r*3));const u=1/(1<<i.fractionalBits),f=zr(i.shDegree),d=.28209479177387814;for(let h=0;h<r;h++){if(a)for(let _=0;_<3;_++)l.position[_]=lE(c[h*3+_]);else for(let _=0;_<3;_++){const v=h*9+_*3;let A=i.positions[v];A|=i.positions[v+1]<<8,A|=i.positions[v+2]<<16,A|=A&8388608?4278190080:0,l.position[_]=A*u}for(let _=0;_<3;_++)l.scale[_]=Math.exp(i.scales[h*3+_]/16-10);const x=i.rotations.subarray(h*3,h*3+3),p=[x[0]/127.5-1,x[1]/127.5-1,x[2]/127.5-1];l.rotation[0]=p[0],l.rotation[1]=p[1],l.rotation[2]=p[2];const g=p[0]*p[0]+p[1]*p[1]+p[2]*p[2];l.rotation[3]=Math.sqrt(Math.max(0,1-g)),l.alpha=Math.floor(i.alphas[h]);for(let _=0;_<3;_++)l.color[_]=Math.floor(((i.colors[h*3+_]/255-.5)/aE*d+.5)*255);for(let _=0;_<3;_++)for(let v=0;v<f;v++)l.sh[_*f+v]=cE(i.sh[f*3*h+v*3+_]);const m=uE(l,i.shDegree,e);if(t){const _=Z.CompressionLevels[0].SphericalHarmonicsDegrees[e].BytesPerSplat,v=h*_+s;Z.writeSplatDataToSectionBuffer(m,n,v,0,e)}else n.addSplat(m)}}const dE=16,hE=1e7;function pE(i){const e=new DataView(i);let t=0;const n={magic:e.getUint32(t,!0),version:e.getUint32(t+4,!0),numPoints:e.getUint32(t+8,!0),shDegree:e.getUint8(t+12),fractionalBits:e.getUint8(t+13),flags:e.getUint8(t+14),reserved:e.getUint8(t+15)};if(t+=dE,n.magic!==rE)return console.error("[SPZ ERROR] deserializePackedGaussians: header not found"),null;if(n.version<1||n.version>2)return console.error(`[SPZ ERROR] deserializePackedGaussians: version not supported: ${n.version}`),null;if(n.numPoints>hE)return console.error(`[SPZ ERROR] deserializePackedGaussians: Too many points: ${n.numPoints}`),null;if(n.shDegree>3)return console.error(`[SPZ ERROR] deserializePackedGaussians: Unsupported SH degree: ${n.shDegree}`),null;const s=n.numPoints,r=zr(n.shDegree),o=n.version===1,a={numPoints:s,shDegree:n.shDegree,fractionalBits:n.fractionalBits,antialiased:(n.flags&oE)!==0,positions:new Uint8Array(s*3*(o?2:3)),scales:new Uint8Array(s*3),rotations:new Uint8Array(s*3),alphas:new Uint8Array(s),colors:new Uint8Array(s*3),sh:new Uint8Array(s*r*3)};try{const l=new Uint8Array(i);let c=a.positions.length,u=t;if(a.positions.set(l.slice(u,u+c)),u+=c,a.alphas.set(l.slice(u,u+a.alphas.length)),u+=a.alphas.length,a.colors.set(l.slice(u,u+a.colors.length)),u+=a.colors.length,a.scales.set(l.slice(u,u+a.scales.length)),u+=a.scales.length,a.rotations.set(l.slice(u,u+a.rotations.length)),u+=a.rotations.length,a.sh.set(l.slice(u,u+a.sh.length)),u+a.sh.length!==i.byteLength)return console.error("[SPZ ERROR] deserializePackedGaussians: incorrect buffer size"),null}catch(l){return console.error("[SPZ ERROR] deserializePackedGaussians: read error",l),null}return a}async function mE(i){try{const e=await iE(i);return pE(e.buffer)}catch(e){return console.error("[SPZ ERROR] loadSpzPacked: decompression error",e),null}}class _h{static loadFromURL(e,t,n,s,r=!0,o=0,a,l,c,u,f){return t&&t(0,"0%",an.Downloading),Qc(e,t,!0,a).then(d=>(t&&t(0,"0%",an.Processing),_h.loadFromFileData(d,n,s,r,o,l,c,u,f)))}static async loadFromFileData(e,t,n,s,r=0,o,a,l,c){await pi();const u=await mE(e);r=Math.min(u.shDegree,r);const f=new De(r);if(s)return zm(u,r,!1,f,0),dl.getStandardGenerator(t,n,o,a,l,c).generateFromUncompressedSplatArray(f);{const{splatBuffer:d,splatBufferDataOffsetBytes:h}=Z.preallocateUncompressed(u.numPoints,r);return zm(u,r,!0,d.bufferData,h),d}}}class Bt{static RowSizeBytes=32;static CenterSizeBytes=12;static ScaleSizeBytes=12;static RotationSizeBytes=4;static ColorSizeBytes=4;static parseToUncompressedSplatBufferSection(e,t,n,s,r,o){const a=Z.CompressionLevels[0].BytesPerCenter,l=Z.CompressionLevels[0].BytesPerScale,c=Z.CompressionLevels[0].BytesPerRotation,u=Z.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat;for(let f=e;f<=t;f++){const d=f*Bt.RowSizeBytes+s,h=new Float32Array(n,d,3),x=new Float32Array(n,d+Bt.CenterSizeBytes,3),p=new Uint8Array(n,d+Bt.CenterSizeBytes+Bt.ScaleSizeBytes,4),g=new Uint8Array(n,d+Bt.CenterSizeBytes+Bt.ScaleSizeBytes+Bt.RotationSizeBytes,4),m=new Gt((g[1]-128)/128,(g[2]-128)/128,(g[3]-128)/128,(g[0]-128)/128);m.normalize();const _=f*u+o,v=new Float32Array(r,_,3),A=new Float32Array(r,_+a,3),S=new Float32Array(r,_+a+l,4),y=new Uint8Array(r,_+a+l+c,4);v[0]=h[0],v[1]=h[1],v[2]=h[2],A[0]=x[0],A[1]=x[1],A[2]=x[2],S[0]=m.w,S[1]=m.x,S[2]=m.y,S[3]=m.z,y[0]=p[0],y[1]=p[1],y[2]=p[2],y[3]=p[3]}}static parseToUncompressedSplatArraySection(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*Bt.RowSizeBytes+s,l=new Float32Array(n,a,3),c=new Float32Array(n,a+Bt.CenterSizeBytes,3),u=new Uint8Array(n,a+Bt.CenterSizeBytes+Bt.ScaleSizeBytes,4),f=new Uint8Array(n,a+Bt.CenterSizeBytes+Bt.ScaleSizeBytes+Bt.RotationSizeBytes,4),d=new Gt((f[1]-128)/128,(f[2]-128)/128,(f[3]-128)/128,(f[0]-128)/128);d.normalize(),r.addSplatFromComonents(l[0],l[1],l[2],c[0],c[1],c[2],d.w,d.x,d.y,d.z,u[0],u[1],u[2],u[3])}}static parseStandardSplatToUncompressedSplatArray(e){const t=e.byteLength/Bt.RowSizeBytes,n=new De;for(let s=0;s<t;s++){const r=s*Bt.RowSizeBytes,o=new Float32Array(e,r,3),a=new Float32Array(e,r+Bt.CenterSizeBytes,3),l=new Uint8Array(e,r+Bt.CenterSizeBytes+Bt.ScaleSizeBytes,4),c=new Uint8Array(e,r+Bt.CenterSizeBytes+Bt.ScaleSizeBytes+Bt.ColorSizeBytes,4),u=new Gt((c[1]-128)/128,(c[2]-128)/128,(c[3]-128)/128,(c[0]-128)/128);u.normalize(),n.addSplatFromComonents(o[0],o[1],o[2],a[0],a[1],a[2],u.w,u.x,u.y,u.z,l[0],l[1],l[2],l[3])}return n}}function km(i,e,t,n,s,r,o,a){return e?dl.getStandardGenerator(t,n,s,r,o,a).generateFromUncompressedSplatArray(i):Z.generateFromUncompressedSplatArrays([i],t,0,new U)}class vh{static loadFromURL(e,t,n,s,r,o,a=!0,l,c,u,f,d){let h=n?qt.ProgressiveToSplatBuffer:qt.ProgressiveToSplatArray;a&&(h=qt.ProgressiveToSplatArray);const x=Z.HeaderSizeBytes+Z.SectionHeaderSizeBytes,p=Ot.ProgressiveLoadSectionSize,g=1;let m,_,v,A=0,S=0,y;const M=oh();let E=0,b=0,C=[];const P=(B,N,k,V)=>{const F=B>=100;if(k&&C.push(k),h===qt.DownloadBeforeProcessing){F&&M.resolve(C);return}if(!V){if(n)throw new bc("Cannon directly load .splat because no file size info is available.");h=qt.DownloadBeforeProcessing;return}if(!m){A=V/Bt.RowSizeBytes,m=new ArrayBuffer(V);const q=Z.CompressionLevels[0].SphericalHarmonicsDegrees[0].BytesPerSplat,G=x+q*A;h===qt.ProgressiveToSplatBuffer?(_=new ArrayBuffer(G),Z.writeHeaderToBuffer({versionMajor:Z.CurrentMajorVersion,versionMinor:Z.CurrentMinorVersion,maxSectionCount:g,sectionCount:g,maxSplatCount:A,splatCount:S,compressionLevel:0,sceneCenter:new U},_)):y=new De(0)}if(k){new Uint8Array(m,b,k.byteLength).set(new Uint8Array(k)),b+=k.byteLength;const q=b-E;if(q>p||F){const j=(F?q:p)/Bt.RowSizeBytes,ce=S+j;h===qt.ProgressiveToSplatBuffer?Bt.parseToUncompressedSplatBufferSection(S,ce-1,m,0,_,x):Bt.parseToUncompressedSplatArraySection(S,ce-1,m,0,y),S=ce,h===qt.ProgressiveToSplatBuffer&&(v||(Z.writeSectionHeaderToBuffer({maxSplatCount:A,splatCount:S,bucketSize:0,bucketCount:0,bucketBlockSize:0,compressionScaleRange:0,storageSizeBytes:0,fullBucketCount:0,partiallyFilledBucketCount:0},0,_,Z.HeaderSizeBytes),v=new Z(_,!1)),v.updateLoadedCounts(1,S),s&&s(v,F)),E+=p}}F&&(h===qt.ProgressiveToSplatBuffer?M.resolve(v):M.resolve(y)),t&&t(B,N,an.Downloading)};return t&&t(0,"0%",an.Downloading),Qc(e,P,!1,l).then(()=>(t&&t(0,"0%",an.Processing),M.promise.then(B=>(t&&t(100,"100%",an.Done),h===qt.DownloadBeforeProcessing?new Blob(C).arrayBuffer().then(N=>vh.loadFromFileData(N,r,o,a,c,u,f,d)):h===qt.ProgressiveToSplatBuffer?B:pi(()=>km(B,a,r,o,c,u,f,d))))))}static loadFromFileData(e,t,n,s,r,o,a,l){return pi(()=>{const c=Bt.parseStandardSplatToUncompressedSplatArray(e);return km(c,s,t,n,r,o,a,l)})}}class Fa{static checkVersion(e){const t=Z.CurrentMajorVersion,n=Z.CurrentMinorVersion,s=Z.parseHeader(e);if(s.versionMajor===t&&s.versionMinor>=n||s.versionMajor>t)return!0;throw new Error(`KSplat version not supported: v${s.versionMajor}.${s.versionMinor}. Minimum required: v${t}.${n}`)}static loadFromURL(e,t,n,s,r){let o,a,l,c,u=!1,f=!1,d,h=[],x=!1,p=!1,g=0,m=0,_=0,v=!1,A=!1,S=!1,y=[];const M=oh(),E=()=>{!u&&!f&&g>=Z.HeaderSizeBytes&&(f=!0,new Blob(y).arrayBuffer().then(V=>{l=new ArrayBuffer(Z.HeaderSizeBytes),new Uint8Array(l).set(new Uint8Array(V,0,Z.HeaderSizeBytes)),Fa.checkVersion(l),f=!1,u=!0,c=Z.parseHeader(l),window.setTimeout(()=>{P()},1)}))};let b=0;const C=()=>{b===0&&(b++,window.setTimeout(()=>{b--,B()},1))},P=()=>{const k=()=>{p=!0,new Blob(y).arrayBuffer().then(F=>{p=!1,x=!0,d=new ArrayBuffer(c.maxSectionCount*Z.SectionHeaderSizeBytes),new Uint8Array(d).set(new Uint8Array(F,Z.HeaderSizeBytes,c.maxSectionCount*Z.SectionHeaderSizeBytes)),h=Z.parseSectionHeaders(c,d,0,!1);let q=0;for(let j=0;j<c.maxSectionCount;j++)q+=h[j].storageSizeBytes;const G=Z.HeaderSizeBytes+c.maxSectionCount*Z.SectionHeaderSizeBytes+q;if(!o){o=new ArrayBuffer(G);let j=0;for(let ce=0;ce<y.length;ce++){const ye=y[ce];new Uint8Array(o,j,ye.byteLength).set(new Uint8Array(ye)),j+=ye.byteLength}}_=Z.HeaderSizeBytes+Z.SectionHeaderSizeBytes*c.maxSectionCount;for(let j=0;j<=h.length&&j<c.maxSectionCount;j++)_+=h[j].storageSizeBytes;C()})};!p&&!x&&u&&g>=Z.HeaderSizeBytes+Z.SectionHeaderSizeBytes*c.maxSectionCount&&k()},B=()=>{if(S)return;S=!0;const k=()=>{if(S=!1,x){if(A)return;if(v=g>=_,g-m>Ot.ProgressiveLoadSectionSize||v){m+=Ot.ProgressiveLoadSectionSize,A=m>=_,a||(a=new Z(o,!1));const F=Z.HeaderSizeBytes+Z.SectionHeaderSizeBytes*c.maxSectionCount;let q=0,G=0,j=0;for(let Ie=0;Ie<c.maxSectionCount;Ie++){const Ne=h[Ie],de=q+Ne.partiallyFilledBucketCount*4+Ne.bucketStorageSizeBytes*Ne.bucketCount,qe=F+de;if(m>=qe){G++;const ee=m-qe,Ge=Z.CompressionLevels[c.compressionLevel].SphericalHarmonicsDegrees[Ne.sphericalHarmonicsDegree].BytesPerSplat;let Ce=Math.floor(ee/Ge);Ce=Math.min(Ce,Ne.maxSplatCount),j+=Ce,a.updateLoadedCounts(G,j),a.updateSectionLoadedCounts(Ie,Ce)}else break;q+=Ne.storageSizeBytes}s(a,A);const ce=m/_*100,ye=ce.toFixed(2)+"%";t&&t(ce,ye,an.Downloading),A?M.resolve(a):B()}}};window.setTimeout(k,Ot.ProgressiveLoadSectionDelayDuration)};return Qc(e,(k,V,F)=>{F&&(y.push(F),o&&new Uint8Array(o,g,F.byteLength).set(new Uint8Array(F)),g+=F.byteLength),n?(E(),P(),B()):t&&t(k,V,an.Downloading)},!n,r).then(k=>(t&&t(0,"0%",an.Processing),(n?M.promise:Fa.loadFromFileData(k)).then(F=>(t&&t(100,"100%",an.Done),F))))}static loadFromFileData(e){return pi(()=>(Fa.checkVersion(e),new Z(e)))}static downloadFile=(function(){let e;return function(t,n){const s=new Blob([t.bufferData],{type:"application/octet-stream"});e||(e=document.createElement("a"),document.body.appendChild(e)),e.download=n,e.href=URL.createObjectURL(s),e.click()}})()}const ei={Splat:0,KSplat:1,Ply:2,Spz:3},Hm=i=>i.endsWith(".ply")?ei.Ply:i.endsWith(".splat")?ei.Splat:i.endsWith(".ksplat")?ei.KSplat:i.endsWith(".spz")?ei.Spz:null,Vm={type:"change"},Zu={type:"start"},Gm={type:"end"},Kl=new th,Wm=new Ks,gE=Math.cos(70*An.DEG2RAD);class jl extends Kr{constructor(e,t){super(),this.object=e,this.domElement=t,this.domElement.style.touchAction="none",this.enabled=!0,this.target=new U,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"KeyA",UP:"KeyW",RIGHT:"KeyD",BOTTOM:"KeyS"},this.mouseButtons={LEFT:eo.ROTATE,MIDDLE:eo.DOLLY,RIGHT:eo.PAN},this.touches={ONE:to.ROTATE,TWO:to.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this.getPolarAngle=function(){return a.phi},this.getAzimuthalAngle=function(){return a.theta},this.getDistance=function(){return this.object.position.distanceTo(this.target)},this.listenToKeyEvents=function(H){H.addEventListener("keydown",T),this._domElementKeyEvents=H},this.stopListenToKeyEvents=function(){this._domElementKeyEvents.removeEventListener("keydown",T),this._domElementKeyEvents=null},this.saveState=function(){n.target0.copy(n.target),n.position0.copy(n.object.position),n.zoom0=n.object.zoom},this.reset=function(){n.target.copy(n.target0),n.object.position.copy(n.position0),n.object.zoom=n.zoom0,this.clearDampedRotation(),this.clearDampedPan(),n.object.updateProjectionMatrix(),n.dispatchEvent(Vm),n.update(),r=s.NONE},this.clearDampedRotation=function(){l.theta=0,l.phi=0},this.clearDampedPan=function(){u.set(0,0,0)},this.update=(function(){const H=new U,J=new Gt().setFromUnitVectors(e.up,new U(0,1,0)),_e=J.clone().invert(),we=new U,Le=new Gt,Re=new U,$e=2*Math.PI;return function(){J.setFromUnitVectors(e.up,new U(0,1,0)),_e.copy(J).invert();const Fe=n.object.position;H.copy(Fe).sub(n.target),H.applyQuaternion(J),a.setFromVector3(H),n.autoRotate&&r===s.NONE&&P(b()),n.enableDamping?(a.theta+=l.theta*n.dampingFactor,a.phi+=l.phi*n.dampingFactor):(a.theta+=l.theta,a.phi+=l.phi);let Ae=n.minAzimuthAngle,xe=n.maxAzimuthAngle;isFinite(Ae)&&isFinite(xe)&&(Ae<-Math.PI?Ae+=$e:Ae>Math.PI&&(Ae-=$e),xe<-Math.PI?xe+=$e:xe>Math.PI&&(xe-=$e),Ae<=xe?a.theta=Math.max(Ae,Math.min(xe,a.theta)):a.theta=a.theta>(Ae+xe)/2?Math.max(Ae,a.theta):Math.min(xe,a.theta)),a.phi=Math.max(n.minPolarAngle,Math.min(n.maxPolarAngle,a.phi)),a.makeSafe(),n.enableDamping===!0?n.target.addScaledVector(u,n.dampingFactor):n.target.add(u),n.zoomToCursor&&y||n.object.isOrthographicCamera?a.radius=j(a.radius):a.radius=j(a.radius*c),H.setFromSpherical(a),H.applyQuaternion(_e),Fe.copy(n.target).add(H),n.object.lookAt(n.target),n.enableDamping===!0?(l.theta*=1-n.dampingFactor,l.phi*=1-n.dampingFactor,u.multiplyScalar(1-n.dampingFactor)):(l.set(0,0,0),u.set(0,0,0));let me=!1;if(n.zoomToCursor&&y){let ge=null;if(n.object.isPerspectiveCamera){const We=H.length();ge=j(We*c);const et=We-ge;n.object.position.addScaledVector(A,et),n.object.updateMatrixWorld()}else if(n.object.isOrthographicCamera){const We=new U(S.x,S.y,0);We.unproject(n.object),n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),me=!0;const et=new U(S.x,S.y,0);et.unproject(n.object),n.object.position.sub(et).add(We),n.object.updateMatrixWorld(),ge=H.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),n.zoomToCursor=!1;ge!==null&&(this.screenSpacePanning?n.target.set(0,0,-1).transformDirection(n.object.matrix).multiplyScalar(ge).add(n.object.position):(Kl.origin.copy(n.object.position),Kl.direction.set(0,0,-1).transformDirection(n.object.matrix),Math.abs(n.object.up.dot(Kl.direction))<gE?e.lookAt(n.target):(Wm.setFromNormalAndCoplanarPoint(n.object.up,n.target),Kl.intersectPlane(Wm,n.target))))}else n.object.isOrthographicCamera&&(n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/c)),n.object.updateProjectionMatrix(),me=!0);return c=1,y=!1,me||we.distanceToSquared(n.object.position)>o||8*(1-Le.dot(n.object.quaternion))>o||Re.distanceToSquared(n.target)>0?(n.dispatchEvent(Vm),we.copy(n.object.position),Le.copy(n.object.quaternion),Re.copy(n.target),me=!1,!0):!1}})(),this.dispose=function(){n.domElement.removeEventListener("contextmenu",fe),n.domElement.removeEventListener("pointerdown",he),n.domElement.removeEventListener("pointercancel",pe),n.domElement.removeEventListener("wheel",R),n.domElement.removeEventListener("pointermove",ie),n.domElement.removeEventListener("pointerup",pe),n._domElementKeyEvents!==null&&(n._domElementKeyEvents.removeEventListener("keydown",T),n._domElementKeyEvents=null)};const n=this,s={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6};let r=s.NONE;const o=1e-6,a=new Zp,l=new Zp;let c=1;const u=new U,f=new Ke,d=new Ke,h=new Ke,x=new Ke,p=new Ke,g=new Ke,m=new Ke,_=new Ke,v=new Ke,A=new U,S=new Ke;let y=!1;const M=[],E={};function b(){return 2*Math.PI/60/60*n.autoRotateSpeed}function C(){return Math.pow(.95,n.zoomSpeed)}function P(H){l.theta-=H}function B(H){l.phi-=H}const N=(function(){const H=new U;return function(_e,we){H.setFromMatrixColumn(we,0),H.multiplyScalar(-_e),u.add(H)}})(),k=(function(){const H=new U;return function(_e,we){n.screenSpacePanning===!0?H.setFromMatrixColumn(we,1):(H.setFromMatrixColumn(we,0),H.crossVectors(n.object.up,H)),H.multiplyScalar(_e),u.add(H)}})(),V=(function(){const H=new U;return function(_e,we){const Le=n.domElement;if(n.object.isPerspectiveCamera){const Re=n.object.position;H.copy(Re).sub(n.target);let $e=H.length();$e*=Math.tan(n.object.fov/2*Math.PI/180),N(2*_e*$e/Le.clientHeight,n.object.matrix),k(2*we*$e/Le.clientHeight,n.object.matrix)}else n.object.isOrthographicCamera?(N(_e*(n.object.right-n.object.left)/n.object.zoom/Le.clientWidth,n.object.matrix),k(we*(n.object.top-n.object.bottom)/n.object.zoom/Le.clientHeight,n.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),n.enablePan=!1)}})();function F(H){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c/=H:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function q(H){n.object.isPerspectiveCamera||n.object.isOrthographicCamera?c*=H:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function G(H){if(!n.zoomToCursor)return;y=!0;const J=n.domElement.getBoundingClientRect(),_e=H.clientX-J.left,we=H.clientY-J.top,Le=J.width,Re=J.height;S.x=_e/Le*2-1,S.y=-(we/Re)*2+1,A.set(S.x,S.y,1).unproject(e).sub(e.position).normalize()}function j(H){return Math.max(n.minDistance,Math.min(n.maxDistance,H))}function ce(H){f.set(H.clientX,H.clientY)}function ye(H){G(H),m.set(H.clientX,H.clientY)}function Ie(H){x.set(H.clientX,H.clientY)}function Ne(H){d.set(H.clientX,H.clientY),h.subVectors(d,f).multiplyScalar(n.rotateSpeed);const J=n.domElement;P(2*Math.PI*h.x/J.clientHeight),B(2*Math.PI*h.y/J.clientHeight),f.copy(d),n.update()}function de(H){_.set(H.clientX,H.clientY),v.subVectors(_,m),v.y>0?F(C()):v.y<0&&q(C()),m.copy(_),n.update()}function qe(H){p.set(H.clientX,H.clientY),g.subVectors(p,x).multiplyScalar(n.panSpeed),V(g.x,g.y),x.copy(p),n.update()}function ee(H){G(H),H.deltaY<0?q(C()):H.deltaY>0&&F(C()),n.update()}function ue(H){let J=!1;switch(H.code){case n.keys.UP:H.ctrlKey||H.metaKey||H.shiftKey?B(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):V(0,n.keyPanSpeed),J=!0;break;case n.keys.BOTTOM:H.ctrlKey||H.metaKey||H.shiftKey?B(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):V(0,-n.keyPanSpeed),J=!0;break;case n.keys.LEFT:H.ctrlKey||H.metaKey||H.shiftKey?P(2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):V(n.keyPanSpeed,0),J=!0;break;case n.keys.RIGHT:H.ctrlKey||H.metaKey||H.shiftKey?P(-2*Math.PI*n.rotateSpeed/n.domElement.clientHeight):V(-n.keyPanSpeed,0),J=!0;break}J&&(H.preventDefault(),n.update())}function Me(){if(M.length===1)f.set(M[0].pageX,M[0].pageY);else{const H=.5*(M[0].pageX+M[1].pageX),J=.5*(M[0].pageY+M[1].pageY);f.set(H,J)}}function Ge(){if(M.length===1)x.set(M[0].pageX,M[0].pageY);else{const H=.5*(M[0].pageX+M[1].pageX),J=.5*(M[0].pageY+M[1].pageY);x.set(H,J)}}function Ce(){const H=M[0].pageX-M[1].pageX,J=M[0].pageY-M[1].pageY,_e=Math.sqrt(H*H+J*J);m.set(0,_e)}function Ye(){n.enableZoom&&Ce(),n.enablePan&&Ge()}function O(){n.enableZoom&&Ce(),n.enableRotate&&Me()}function z(H){if(M.length==1)d.set(H.pageX,H.pageY);else{const _e=Xe(H),we=.5*(H.pageX+_e.x),Le=.5*(H.pageY+_e.y);d.set(we,Le)}h.subVectors(d,f).multiplyScalar(n.rotateSpeed);const J=n.domElement;P(2*Math.PI*h.x/J.clientHeight),B(2*Math.PI*h.y/J.clientHeight),f.copy(d)}function Q(H){if(M.length===1)p.set(H.pageX,H.pageY);else{const J=Xe(H),_e=.5*(H.pageX+J.x),we=.5*(H.pageY+J.y);p.set(_e,we)}g.subVectors(p,x).multiplyScalar(n.panSpeed),V(g.x,g.y),x.copy(p)}function w(H){const J=Xe(H),_e=H.pageX-J.x,we=H.pageY-J.y,Le=Math.sqrt(_e*_e+we*we);_.set(0,Le),v.set(0,Math.pow(_.y/m.y,n.zoomSpeed)),F(v.y),m.copy(_)}function ne(H){n.enableZoom&&w(H),n.enablePan&&Q(H)}function oe(H){n.enableZoom&&w(H),n.enableRotate&&z(H)}function he(H){n.enabled!==!1&&(M.length===0&&(n.domElement.setPointerCapture(H.pointerId),n.domElement.addEventListener("pointermove",ie),n.domElement.addEventListener("pointerup",pe)),$(H),H.pointerType==="touch"?X(H):te(H))}function ie(H){n.enabled!==!1&&(H.pointerType==="touch"?se(H):ve(H))}function pe(H){Be(H),M.length===0&&(n.domElement.releasePointerCapture(H.pointerId),n.domElement.removeEventListener("pointermove",ie),n.domElement.removeEventListener("pointerup",pe)),n.dispatchEvent(Gm),r=s.NONE}function te(H){let J;switch(H.button){case 0:J=n.mouseButtons.LEFT;break;case 1:J=n.mouseButtons.MIDDLE;break;case 2:J=n.mouseButtons.RIGHT;break;default:J=-1}switch(J){case eo.DOLLY:if(n.enableZoom===!1)return;ye(H),r=s.DOLLY;break;case eo.ROTATE:if(H.ctrlKey||H.metaKey||H.shiftKey){if(n.enablePan===!1)return;Ie(H),r=s.PAN}else{if(n.enableRotate===!1)return;ce(H),r=s.ROTATE}break;case eo.PAN:if(H.ctrlKey||H.metaKey||H.shiftKey){if(n.enableRotate===!1)return;ce(H),r=s.ROTATE}else{if(n.enablePan===!1)return;Ie(H),r=s.PAN}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(Zu)}function ve(H){switch(r){case s.ROTATE:if(n.enableRotate===!1)return;Ne(H);break;case s.DOLLY:if(n.enableZoom===!1)return;de(H);break;case s.PAN:if(n.enablePan===!1)return;qe(H);break}}function R(H){n.enabled===!1||n.enableZoom===!1||r!==s.NONE||(H.preventDefault(),n.dispatchEvent(Zu),ee(H),n.dispatchEvent(Gm))}function T(H){n.enabled===!1||n.enablePan===!1||ue(H)}function X(H){switch(be(H),M.length){case 1:switch(n.touches.ONE){case to.ROTATE:if(n.enableRotate===!1)return;Me(),r=s.TOUCH_ROTATE;break;case to.PAN:if(n.enablePan===!1)return;Ge(),r=s.TOUCH_PAN;break;default:r=s.NONE}break;case 2:switch(n.touches.TWO){case to.DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;Ye(),r=s.TOUCH_DOLLY_PAN;break;case to.DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;O(),r=s.TOUCH_DOLLY_ROTATE;break;default:r=s.NONE}break;default:r=s.NONE}r!==s.NONE&&n.dispatchEvent(Zu)}function se(H){switch(be(H),r){case s.TOUCH_ROTATE:if(n.enableRotate===!1)return;z(H),n.update();break;case s.TOUCH_PAN:if(n.enablePan===!1)return;Q(H),n.update();break;case s.TOUCH_DOLLY_PAN:if(n.enableZoom===!1&&n.enablePan===!1)return;ne(H),n.update();break;case s.TOUCH_DOLLY_ROTATE:if(n.enableZoom===!1&&n.enableRotate===!1)return;oe(H),n.update();break;default:r=s.NONE}}function fe(H){n.enabled!==!1&&H.preventDefault()}function $(H){M.push(H)}function Be(H){delete E[H.pointerId];for(let J=0;J<M.length;J++)if(M[J].pointerId==H.pointerId){M.splice(J,1);return}}function be(H){let J=E[H.pointerId];J===void 0&&(J=new Ke,E[H.pointerId]=J),J.set(H.pageX,H.pageY)}function Xe(H){const J=H.pointerId===M[0].pointerId?M[1]:M[0];return E[J.pointerId]}n.domElement.addEventListener("contextmenu",fe),n.domElement.addEventListener("pointerdown",he),n.domElement.addEventListener("pointercancel",pe),n.domElement.addEventListener("wheel",R,{passive:!1}),this.update()}}const xE=(i,e,t,n,s)=>{const r=performance.now();let o=i.style.display==="none"?0:parseFloat(i.style.opacity);isNaN(o)&&(o=1);const a=window.setInterval(()=>{const c=performance.now()-r;let u=Math.min(c/n,1);u>.999&&(u=1);let f;e?(f=(1-u)*o,f<1e-4&&(f=0)):f=(1-o)*u+o,f>0?(i.style.display=t,i.style.opacity=f):i.style.display="none",u>=1&&(s&&s(),window.clearInterval(a))},16);return a},_E=500;class Ah{static elementIDGen=0;constructor(e,t){this.taskIDGen=0,this.elementID=Ah.elementIDGen++,this.tasks=[],this.message=e||"Loading...",this.container=t||document.body,this.spinnerContainerOuter=document.createElement("div"),this.spinnerContainerOuter.className=`spinnerOuterContainer${this.elementID}`,this.spinnerContainerOuter.style.display="none",this.spinnerContainerPrimary=document.createElement("div"),this.spinnerContainerPrimary.className=`spinnerContainerPrimary${this.elementID}`,this.spinnerPrimary=document.createElement("div"),this.spinnerPrimary.classList.add(`spinner${this.elementID}`,`spinnerPrimary${this.elementID}`),this.messageContainerPrimary=document.createElement("div"),this.messageContainerPrimary.classList.add(`messageContainer${this.elementID}`,`messageContainerPrimary${this.elementID}`),this.messageContainerPrimary.innerHTML=this.message,this.spinnerContainerMin=document.createElement("div"),this.spinnerContainerMin.className=`spinnerContainerMin${this.elementID}`,this.spinnerMin=document.createElement("div"),this.spinnerMin.classList.add(`spinner${this.elementID}`,`spinnerMin${this.elementID}`),this.messageContainerMin=document.createElement("div"),this.messageContainerMin.classList.add(`messageContainer${this.elementID}`,`messageContainerMin${this.elementID}`),this.messageContainerMin.innerHTML=this.message,this.spinnerContainerPrimary.appendChild(this.spinnerPrimary),this.spinnerContainerPrimary.appendChild(this.messageContainerPrimary),this.spinnerContainerOuter.appendChild(this.spinnerContainerPrimary),this.spinnerContainerMin.appendChild(this.spinnerMin),this.spinnerContainerMin.appendChild(this.messageContainerMin),this.spinnerContainerOuter.appendChild(this.spinnerContainerMin);const n=document.createElement("style");n.innerHTML=`

            .spinnerOuterContainer${this.elementID} {
                width: 100%;
                height: 100%;
                margin: 0;
                top: 0;
                left: 0;
                position: absolute;
                pointer-events: none;
            }

            .messageContainer${this.elementID} {
                height: 20px;
                font-family: arial;
                font-size: 12pt;
                color: #ffffff;
                text-align: center;
                vertical-align: middle;
            }

            .spinner${this.elementID} {
                padding: 15px;
                background: #07e8d6;
                z-index:99999;
            
                aspect-ratio: 1;
                border-radius: 50%;
                --_m: 
                    conic-gradient(#0000,#000),
                    linear-gradient(#000 0 0) content-box;
                -webkit-mask: var(--_m);
                    mask: var(--_m);
                -webkit-mask-composite: source-out;
                    mask-composite: subtract;
                box-sizing: border-box;
                animation: load 1s linear infinite;
            }

            .spinnerContainerPrimary${this.elementID} {
                z-index:99999;
                background-color: rgba(128, 128, 128, 0.75);
                border: #666666 1px solid;
                border-radius: 5px;
                padding-top: 20px;
                padding-bottom: 10px;
                margin: 0;
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-80px, -80px);
                width: 180px;
                pointer-events: auto;
            }

            .spinnerPrimary${this.elementID} {
                width: 120px;
                margin-left: 30px;
            }

            .messageContainerPrimary${this.elementID} {
                padding-top: 15px;
            }

            .spinnerContainerMin${this.elementID} {
                z-index:99999;
                background-color: rgba(128, 128, 128, 0.75);
                border: #666666 1px solid;
                border-radius: 5px;
                padding-top: 20px;
                padding-bottom: 15px;
                margin: 0;
                position: absolute;
                bottom: 50px;
                left: 50%;
                transform: translate(-50%, 0);
                display: flex;
                flex-direction: left;
                pointer-events: auto;
                min-width: 250px;
            }

            .messageContainerMin${this.elementID} {
                margin-right: 15px;
            }

            .spinnerMin${this.elementID} {
                width: 50px;
                height: 50px;
                margin-left: 15px;
                margin-right: 25px;
            }

            .messageContainerMin${this.elementID} {
                padding-top: 15px;
            }
            
            @keyframes load {
                to{transform: rotate(1turn)}
            }

        `,this.spinnerContainerOuter.appendChild(n),this.container.appendChild(this.spinnerContainerOuter),this.setMinimized(!1,!0),this.fadeTransitions=[]}addTask(e){const t={message:e,id:this.taskIDGen++};return this.tasks.push(t),this.update(),t.id}removeTask(e){let t=0;for(let n of this.tasks){if(n.id===e){this.tasks.splice(t,1);break}t++}this.update()}removeAllTasks(){this.tasks=[],this.update()}setMessageForTask(e,t){for(let n of this.tasks)if(n.id===e){n.message=t;break}this.update()}update(){this.tasks.length>0?(this.show(),this.setMessage(this.tasks[this.tasks.length-1].message)):this.hide()}show(){this.spinnerContainerOuter.style.display="block",this.visible=!0}hide(){this.spinnerContainerOuter.style.display="none",this.visible=!1}setContainer(e){this.container&&this.spinnerContainerOuter.parentElement===this.container&&this.container.removeChild(this.spinnerContainerOuter),e&&(this.container=e,this.container.appendChild(this.spinnerContainerOuter),this.spinnerContainerOuter.style.zIndex=this.container.style.zIndex+1)}setMinimized(e,t){const n=(s,r,o,a,l)=>{o?s.style.display=r?a:"none":this.fadeTransitions[l]=xE(s,!r,a,_E,()=>{this.fadeTransitions[l]=null})};n(this.spinnerContainerPrimary,!e,t,"block",0),n(this.spinnerContainerMin,e,t,"flex",1),this.minimized=e}setMessage(e){this.messageContainerPrimary.innerHTML=e,this.messageContainerMin.innerHTML=e}}class vE{constructor(e){this.idGen=0,this.tasks=[],this.container=e||document.body,this.progressBarContainerOuter=document.createElement("div"),this.progressBarContainerOuter.className="progressBarOuterContainer",this.progressBarContainerOuter.style.display="none",this.progressBarBox=document.createElement("div"),this.progressBarBox.className="progressBarBox",this.progressBarBackground=document.createElement("div"),this.progressBarBackground.className="progressBarBackground",this.progressBar=document.createElement("div"),this.progressBar.className="progressBar",this.progressBarBackground.appendChild(this.progressBar),this.progressBarBox.appendChild(this.progressBarBackground),this.progressBarContainerOuter.appendChild(this.progressBarBox);const t=document.createElement("style");t.innerHTML=`

            .progressBarOuterContainer {
                width: 100%;
                height: 100%;
                margin: 0;
                top: 0;
                left: 0;
                position: absolute;
                pointer-events: none;
            }

            .progressBarBox {
                z-index:99999;
                padding: 7px 9px 5px 7px;
                background-color: rgba(190, 190, 190, 0.75);
                border: #555555 1px solid;
                border-radius: 15px;
                margin: 0;
                position: absolute;
                bottom: 50px;
                left: 50%;
                transform: translate(-50%, 0);
                width: 180px;
                height: 30px;
                pointer-events: auto;
            }

            .progressBarBackground {
                width: 100%;
                height: 25px;
                border-radius:10px;
                background-color: rgba(128, 128, 128, 0.75);
                border: #444444 1px solid;
                box-shadow: inset 0 0 10px #333333;
            }

            .progressBar {
                height: 25px;
                width: 0px;
                border-radius:10px;
                background-color: rgba(0, 200, 0, 0.75);
                box-shadow: inset 0 0 10px #003300;
            }

        `,this.progressBarContainerOuter.appendChild(t),this.container.appendChild(this.progressBarContainerOuter)}show(){this.progressBarContainerOuter.style.display="block"}hide(){this.progressBarContainerOuter.style.display="none"}setProgress(e){this.progressBar.style.width=e+"%"}setContainer(e){this.container&&this.progressBarContainerOuter.parentElement===this.container&&this.container.removeChild(this.progressBarContainerOuter),e&&(this.container=e,this.container.appendChild(this.progressBarContainerOuter),this.progressBarContainerOuter.style.zIndex=this.container.style.zIndex+1)}}class AE{constructor(e){this.container=e||document.body,this.infoCells={};const t=[["Camera position","cameraPosition"],["Camera look-at","cameraLookAt"],["Camera up","cameraUp"],["Camera mode","orthographicCamera"],["Cursor position","cursorPosition"],["FPS","fps"],["Rendering:","renderSplatCount"],["Sort time","sortTime"],["Render window","renderWindow"],["Focal adjustment","focalAdjustment"],["Splat scale","splatScale"],["Point cloud mode","pointCloudMode"]];this.infoPanelContainer=document.createElement("div");const n=document.createElement("style");n.innerHTML=`

            .infoPanel {
                width: 430px;
                padding: 10px;
                background-color: rgba(50, 50, 50, 0.85);
                border: #555555 2px solid;
                color: #dddddd;
                border-radius: 10px;
                z-index: 9999;
                font-family: arial;
                font-size: 11pt;
                text-align: left;
                margin: 0;
                top: 10px;
                left:10px;
                position: absolute;
                pointer-events: auto;
            }

            .info-panel-cell {
                margin-bottom: 5px;
                padding-bottom: 2px;
            }

            .label-cell {
                font-weight: bold;
                font-size: 12pt;
                width: 140px;
            }

        `,this.infoPanelContainer.append(n),this.infoPanel=document.createElement("div"),this.infoPanel.className="infoPanel";const s=document.createElement("div");s.style.display="table";for(let r of t){const o=document.createElement("div");o.style.display="table-row",o.className="info-panel-row";const a=document.createElement("div");a.style.display="table-cell",a.innerHTML=`${r[0]}: `,a.classList.add("info-panel-cell","label-cell");const l=document.createElement("div");l.style.display="table-cell",l.style.width="10px",l.innerHTML=" ",l.className="info-panel-cell";const c=document.createElement("div");c.style.display="table-cell",c.innerHTML="",c.className="info-panel-cell",this.infoCells[r[1]]=c,o.appendChild(a),o.appendChild(l),o.appendChild(c),s.appendChild(o)}this.infoPanel.appendChild(s),this.infoPanelContainer.append(this.infoPanel),this.infoPanelContainer.style.display="none",this.container.appendChild(this.infoPanelContainer),this.visible=!1}update=function(e,t,n,s,r,o,a,l,c,u,f,d,h,x){const p=`${t.x.toFixed(5)}, ${t.y.toFixed(5)}, ${t.z.toFixed(5)}`;if(this.infoCells.cameraPosition.innerHTML!==p&&(this.infoCells.cameraPosition.innerHTML=p),n){const m=n,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cameraLookAt.innerHTML!==_&&(this.infoCells.cameraLookAt.innerHTML=_)}const g=`${s.x.toFixed(5)}, ${s.y.toFixed(5)}, ${s.z.toFixed(5)}`;if(this.infoCells.cameraUp.innerHTML!==g&&(this.infoCells.cameraUp.innerHTML=g),this.infoCells.orthographicCamera.innerHTML=r?"Orthographic":"Perspective",o){const m=o,_=`${m.x.toFixed(5)}, ${m.y.toFixed(5)}, ${m.z.toFixed(5)}`;this.infoCells.cursorPosition.innerHTML=_}else this.infoCells.cursorPosition.innerHTML="N/A";this.infoCells.fps.innerHTML=a,this.infoCells.renderWindow.innerHTML=`${e.x} x ${e.y}`,this.infoCells.renderSplatCount.innerHTML=`${c} splats out of ${l} (${u.toFixed(2)}%)`,this.infoCells.sortTime.innerHTML=`${f.toFixed(3)} ms`,this.infoCells.focalAdjustment.innerHTML=`${d.toFixed(3)}`,this.infoCells.splatScale.innerHTML=`${h.toFixed(3)}`,this.infoCells.pointCloudMode.innerHTML=`${x}`};setContainer(e){this.container&&this.infoPanelContainer.parentElement===this.container&&this.container.removeChild(this.infoPanelContainer),e&&(this.container=e,this.container.appendChild(this.infoPanelContainer),this.infoPanelContainer.style.zIndex=this.container.style.zIndex+1)}show(){this.infoPanelContainer.style.display="block",this.visible=!0}hide(){this.infoPanelContainer.style.display="none",this.visible=!1}}const Xm=new U;class SE extends dn{constructor(e=new U(0,0,1),t=new U(0,0,0),n=1,s=.1,r=16776960,o=n*.2,a=o*.2){super(),this.type="ArrowHelper";const l=new Ja(s,s,n,32);l.translate(0,n/2,0);const c=new Ja(0,a,o,32);c.translate(0,n,0),this.position.copy(t),this.line=new un(l,new Yr({color:r,toneMapped:!1})),this.line.matrixAutoUpdate=!1,this.add(this.line),this.cone=new un(c,new Yr({color:r,toneMapped:!1})),this.cone.matrixAutoUpdate=!1,this.add(this.cone),this.setDirection(e)}setDirection(e){if(e.y>.99999)this.quaternion.set(0,0,0,1);else if(e.y<-.99999)this.quaternion.set(1,0,0,0);else{Xm.set(e.z,0,-e.x).normalize();const t=Math.acos(e.y);this.quaternion.setFromAxisAngle(Xm,t)}}setColor(e){this.line.material.color.set(e),this.cone.material.color.set(e)}copy(e){return super.copy(e,!1),this.line.copy(e.line),this.cone.copy(e.cone),this}dispose(){this.line.geometry.dispose(),this.line.material.dispose(),this.cone.geometry.dispose(),this.cone.material.dispose()}}class La{constructor(e){this.threeScene=e,this.splatRenderTarget=null,this.renderTargetCopyQuad=null,this.renderTargetCopyCamera=null,this.meshCursor=null,this.focusMarker=null,this.controlPlane=null,this.debugRoot=null,this.secondaryDebugRoot=null}updateSplatRenderTargetForRenderDimensions(e,t){this.destroySplatRendertarget(),this.splatRenderTarget=new cr(e,t,{format:Yn,stencilBuffer:!1,depthBuffer:!0}),this.splatRenderTarget.depthTexture=new nh(e,t),this.splatRenderTarget.depthTexture.format=Wo,this.splatRenderTarget.depthTexture.type=Ii}destroySplatRendertarget(){this.splatRenderTarget&&(this.splatRenderTarget=null)}setupRenderTargetCopyObjects(){const e={sourceColorTexture:{type:"t",value:null},sourceDepthTexture:{type:"t",value:null}},t=new Kn({vertexShader:`
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = vec4( position.xy, 0.0, 1.0 );    
                }
            `,fragmentShader:`
                #include <common>
                #include <packing>
                varying vec2 vUv;
                uniform sampler2D sourceColorTexture;
                uniform sampler2D sourceDepthTexture;
                void main() {
                    vec4 color = texture2D(sourceColorTexture, vUv);
                    float fragDepth = texture2D(sourceDepthTexture, vUv).x;
                    gl_FragDepth = fragDepth;
                    gl_FragColor = vec4(color.rgb, color.a * 2.0);
              }
            `,uniforms:e,depthWrite:!1,depthTest:!1,transparent:!0,blending:wg,blendSrc:Xa,blendSrcAlpha:Xa,blendDst:qa,blendDstAlpha:qa});t.extensions.fragDepth=!0,this.renderTargetCopyQuad=new un(new Yo(2,2),t),this.renderTargetCopyCamera=new rh(-1,1,1,-1,0,1)}destroyRenderTargetCopyObjects(){this.renderTargetCopyQuad&&(yo(this.renderTargetCopyQuad),this.renderTargetCopyQuad=null)}setupMeshCursor(){if(!this.meshCursor){const e=new ih(.5,1.5,32),t=new Yr({color:16777215}),n=new un(e,t);n.rotation.set(0,0,Math.PI),n.position.set(0,1,0);const s=new un(e,t);s.position.set(0,-1,0);const r=new un(e,t);r.rotation.set(0,0,Math.PI/2),r.position.set(1,0,0);const o=new un(e,t);o.rotation.set(0,0,-Math.PI/2),o.position.set(-1,0,0),this.meshCursor=new dn,this.meshCursor.add(n),this.meshCursor.add(s),this.meshCursor.add(r),this.meshCursor.add(o),this.meshCursor.scale.set(.1,.1,.1),this.threeScene.add(this.meshCursor),this.meshCursor.visible=!1}}destroyMeshCursor(){this.meshCursor&&(yo(this.meshCursor),this.threeScene.remove(this.meshCursor),this.meshCursor=null)}setMeshCursorVisibility(e){this.meshCursor.visible=e}getMeschCursorVisibility(){return this.meshCursor.visible}setMeshCursorPosition(e){this.meshCursor.position.copy(e)}positionAndOrientMeshCursor(e,t){this.meshCursor.position.copy(e),this.meshCursor.up.copy(t.up),this.meshCursor.lookAt(t.position)}setupFocusMarker(){if(!this.focusMarker){const e=new yc(.5,32,32),t=La.buildFocusMarkerMaterial();t.depthTest=!1,t.depthWrite=!1,t.transparent=!0,this.focusMarker=new un(e,t)}}destroyFocusMarker(){this.focusMarker&&(yo(this.focusMarker),this.focusMarker=null)}updateFocusMarker=(function(){const e=new U,t=new rt,n=new U;return function(s,r,o){t.copy(r.matrixWorld).invert(),e.copy(s).applyMatrix4(t),e.normalize().multiplyScalar(10),e.applyMatrix4(r.matrixWorld),n.copy(r.position).sub(s);const a=n.length();this.focusMarker.position.copy(s),this.focusMarker.scale.set(a,a,a),this.focusMarker.material.uniforms.realFocusPosition.value.copy(s),this.focusMarker.material.uniforms.viewport.value.copy(o),this.focusMarker.material.uniformsNeedUpdate=!0}})();setFocusMarkerVisibility(e){this.focusMarker.visible=e}setFocusMarkerOpacity(e){this.focusMarker.material.uniforms.opacity.value=e,this.focusMarker.material.uniformsNeedUpdate=!0}getFocusMarkerOpacity(){return this.focusMarker.material.uniforms.opacity.value}setupControlPlane(){if(!this.controlPlane){const e=new Yo(1,1);e.rotateX(-Math.PI/2);const t=new Yr({color:16777215});t.transparent=!0,t.opacity=.6,t.depthTest=!1,t.depthWrite=!1,t.side=Ei;const n=new un(e,t),s=new U(0,1,0);s.normalize();const r=new U(0,0,0),o=.5,a=.01,l=56576,c=new SE(s,r,o,a,l,.1,.03);this.controlPlane=new dn,this.controlPlane.add(n),this.controlPlane.add(c)}}destroyControlPlane(){this.controlPlane&&(yo(this.controlPlane),this.controlPlane=null)}setControlPlaneVisibility(e){this.controlPlane.visible=e}positionAndOrientControlPlane=(function(){const e=new Gt,t=new U(0,1,0);return function(n,s){e.setFromUnitVectors(t,s),this.controlPlane.position.copy(n),this.controlPlane.quaternion.copy(e)}})();addDebugMeshes(){this.debugRoot=this.createDebugMeshes(),this.secondaryDebugRoot=this.createSecondaryDebugMeshes(),this.threeScene.add(this.debugRoot),this.threeScene.add(this.secondaryDebugRoot)}destroyDebugMeshes(){for(let e of[this.debugRoot,this.secondaryDebugRoot])e&&(yo(e),this.threeScene.remove(e));this.debugRoot=null,this.secondaryDebugRoot=null}createDebugMeshes(e){const t=new yc(1,32,32),n=new dn,s=(r,o)=>{let a=new un(t,La.buildDebugMaterial(r));a.renderOrder=e,n.add(a),a.position.fromArray(o)};return s(16711680,[-50,0,0]),s(16711680,[50,0,0]),s(65280,[0,0,-50]),s(65280,[0,0,50]),s(16755200,[5,0,5]),n}createSecondaryDebugMeshes(e){const t=new na(3,3,3),n=new dn;let s=12303291;const r=a=>{let l=new un(t,La.buildDebugMaterial(s));l.renderOrder=e,n.add(l),l.position.fromArray(a)};let o=10;return r([-o,0,-o]),r([-o,0,o]),r([o,0,-o]),r([o,0,o]),n}static buildDebugMaterial(e){const t=`
            #include <common>
            varying float ndcDepth;

            void main() {
                gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position.xyz, 1.0);
                ndcDepth = gl_Position.z / gl_Position.w;
                gl_Position.x = gl_Position.x / gl_Position.w;
                gl_Position.y = gl_Position.y / gl_Position.w;
                gl_Position.z = 0.0;
                gl_Position.w = 1.0;
    
            }
        `,n=`
            #include <common>
            uniform vec3 color;
            varying float ndcDepth;
            void main() {
                gl_FragDepth = (ndcDepth + 1.0) / 2.0;
                gl_FragColor = vec4(color.rgb, 0.0);
            }
        `,s={color:{type:"v3",value:new vt(e)}},r=new Kn({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!1,depthTest:!0,depthWrite:!0,side:as});return r.extensions.fragDepth=!0,r}static buildFocusMarkerMaterial(e){const t=`
            #include <common>

            uniform vec2 viewport;
            uniform vec3 realFocusPosition;

            varying vec4 ndcPosition;
            varying vec4 ndcCenter;
            varying vec4 ndcFocusPosition;

            void main() {
                float radius = 0.01;

                vec4 viewPosition = modelViewMatrix * vec4(position.xyz, 1.0);
                vec4 viewCenter = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);

                vec4 viewFocusPosition = modelViewMatrix * vec4(realFocusPosition, 1.0);

                ndcPosition = projectionMatrix * viewPosition;
                ndcPosition = ndcPosition * vec4(1.0 / ndcPosition.w);
                ndcCenter = projectionMatrix * viewCenter;
                ndcCenter = ndcCenter * vec4(1.0 / ndcCenter.w);

                ndcFocusPosition = projectionMatrix * viewFocusPosition;
                ndcFocusPosition = ndcFocusPosition * vec4(1.0 / ndcFocusPosition.w);

                gl_Position = projectionMatrix * viewPosition;

            }
        `,n=`
            #include <common>
            uniform vec3 color;
            uniform vec2 viewport;
            uniform float opacity;

            varying vec4 ndcPosition;
            varying vec4 ndcCenter;
            varying vec4 ndcFocusPosition;

            void main() {
                vec2 screenPosition = vec2(ndcPosition) * viewport;
                vec2 screenCenter = vec2(ndcCenter) * viewport;

                vec2 screenVec = screenPosition - screenCenter;

                float projectedRadius = length(screenVec);

                float lineWidth = 0.0005 * viewport.y;
                float aaRange = 0.0025 * viewport.y;
                float radius = 0.06 * viewport.y;
                float radDiff = abs(projectedRadius - radius) - lineWidth;
                float alpha = 1.0 - clamp(radDiff / 5.0, 0.0, 1.0); 

                gl_FragColor = vec4(color.rgb, alpha * opacity);
            }
        `,s={color:{type:"v3",value:new vt(e)},realFocusPosition:{type:"v3",value:new U},viewport:{type:"v2",value:new Ke},opacity:{value:0}};return new Kn({uniforms:s,vertexShader:t,fragmentShader:n,transparent:!0,depthTest:!1,depthWrite:!1,side:as})}dispose(){this.destroyMeshCursor(),this.destroyFocusMarker(),this.destroyDebugMeshes(),this.destroyControlPlane(),this.destroyRenderTargetCopyObjects(),this.destroySplatRendertarget()}}const yE=new U(1,0,0),bE=new U(0,1,0),ME=new U(0,0,1);class Ju{constructor(e=new U,t=new U){this.origin=new U,this.direction=new U,this.setParameters(e,t)}setParameters(e,t){this.origin.copy(e),this.direction.copy(t).normalize()}boxContainsPoint(e,t,n){return!(t.x<e.min.x-n||t.x>e.max.x+n||t.y<e.min.y-n||t.y>e.max.y+n||t.z<e.min.z-n||t.z>e.max.z+n)}intersectBox=(function(){const e=new U,t=[],n=[],s=[];return function(r,o){if(n[0]=this.origin.x,n[1]=this.origin.y,n[2]=this.origin.z,s[0]=this.direction.x,s[1]=this.direction.y,s[2]=this.direction.z,this.boxContainsPoint(r,this.origin,1e-4))return o&&(o.origin.copy(this.origin),o.normal.set(0,0,0),o.distance=-1),!0;for(let a=0;a<3;a++){if(s[a]==0)continue;const l=a==0?yE:a==1?bE:ME,c=s[a]<0?r.max:r.min;let u=-Math.sign(s[a]);t[0]=a==0?c.x:a==1?c.y:c.z;let f=t[0]-n[a];if(f*u<0){const d=(a+1)%3,h=(a+2)%3;if(t[2]=s[d]/s[a]*f+n[d],t[1]=s[h]/s[a]*f+n[h],e.set(t[a],t[h],t[d]),this.boxContainsPoint(r,e,1e-4))return o&&(o.origin.copy(e),o.normal.copy(l).multiplyScalar(u),o.distance=e.sub(this.origin).length()),!0}}return!1}})();intersectSphere=(function(){const e=new U;return function(t,n,s){e.copy(t).sub(this.origin);const r=e.dot(this.direction),o=r*r,l=e.dot(e)-o,c=n*n;if(l>c)return!1;const u=Math.sqrt(c-l),f=r-u,d=r+u;if(d<0)return!1;let h=f<0?d:f;return s&&(s.origin.copy(this.origin).addScaledVector(this.direction,h),s.normal.copy(s.origin).sub(t).normalize(),s.distance=h),!0}})()}class Sh{constructor(){this.origin=new U,this.normal=new U,this.distance=0,this.splatIndex=0}set(e,t,n,s){this.origin.copy(e),this.normal.copy(t),this.distance=n,this.splatIndex=s}clone(){const e=new Sh;return e.origin.copy(this.origin),e.normal.copy(this.normal),e.distance=this.distance,e.splatIndex=this.splatIndex,e}}const Cs={ThreeD:0,TwoD:1};class CE{constructor(e,t,n=!1){this.ray=new Ju(e,t),this.raycastAgainstTrueSplatEllipsoid=n}setFromCameraAndScreenPosition=(function(){const e=new Ke;return function(t,n,s){if(e.x=n.x/s.x*2-1,e.y=(s.y-n.y)/s.y*2-1,t.isPerspectiveCamera)this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t;else if(t.isOrthographicCamera)this.ray.origin.set(e.x,e.y,(t.near+t.far)/(t.near-t.far)).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t;else throw new Error("Raycaster::setFromCameraAndScreenPosition() -> Unsupported camera type")}})();intersectSplatMesh=(function(){const e=new rt,t=new rt,n=new rt,s=new Ju,r=new U;return function(o,a=[]){const l=o.getSplatTree();if(l){for(let c=0;c<l.subTrees.length;c++){const u=l.subTrees[c];t.copy(o.matrixWorld),o.dynamicMode&&(o.getSceneTransform(c,n),t.multiply(n)),e.copy(t).invert(),s.origin.copy(this.ray.origin).applyMatrix4(e),s.direction.copy(this.ray.origin).add(this.ray.direction),s.direction.applyMatrix4(e).sub(s.origin).normalize();const f=[];u.rootNode&&this.castRayAtSplatTreeNode(s,l,u.rootNode,f),f.forEach(d=>{d.origin.applyMatrix4(t),d.normal.applyMatrix4(t).normalize(),d.distance=r.copy(d.origin).sub(this.ray.origin).length()}),a.push(...f)}return a.sort((c,u)=>c.distance>u.distance?1:-1),a}}})();castRayAtSplatTreeNode=(function(){const e=new Zt,t=new U,n=new U,s=new Gt,r=new Sh,o=1e-7,a=new U(0,0,0),l=new rt,c=new rt,u=new rt,f=new rt,d=new rt,h=new Ju;return function(x,p,g,m=[]){if(x.intersectBox(g.boundingBox)){if(g.data&&g.data.indexes&&g.data.indexes.length>0)for(let _=0;_<g.data.indexes.length;_++){const v=g.data.indexes[_],A=p.splatMesh.getSceneIndexForSplat(v);if(p.splatMesh.getScene(A).visible&&(p.splatMesh.getSplatColor(v,e),p.splatMesh.getSplatCenter(v,t),p.splatMesh.getSplatScaleAndRotation(v,n,s),!(n.x<=o||n.y<=o||p.splatMesh.splatRenderMode===Cs.ThreeD&&n.z<=o)))if(this.raycastAgainstTrueSplatEllipsoid){c.makeScale(n.x,n.y,n.z),u.makeRotationFromQuaternion(s);const y=Math.log10(e.w)*2;if(l.makeScale(y,y,y),d.copy(l).multiply(u).multiply(c),f.copy(d).invert(),h.origin.copy(x.origin).sub(t).applyMatrix4(f),h.direction.copy(x.origin).add(x.direction).sub(t),h.direction.applyMatrix4(f).sub(h.origin).normalize(),h.intersectSphere(a,1,r)){const M=r.clone();M.splatIndex=v,M.origin.applyMatrix4(d).add(t),m.push(M)}}else{let y=n.x+n.y,M=2;if(p.splatMesh.splatRenderMode===Cs.ThreeD&&(y+=n.z,M=3),y=y/M,x.intersectSphere(t,y,r)){const E=r.clone();E.splatIndex=v,m.push(E)}}}if(g.children&&g.children.length>0)for(let _ of g.children)this.castRayAtSplatTreeNode(x,p,_,m);return m}}})()}class Lo{static buildVertexShaderBase(e=!1,t=!1,n=0,s=""){let r=`
        precision highp float;
        #include <common>

        attribute uint splatIndex;
        uniform highp usampler2D centersColorsTexture;
        uniform highp sampler2D sphericalHarmonicsTexture;
        uniform highp sampler2D sphericalHarmonicsTextureR;
        uniform highp sampler2D sphericalHarmonicsTextureG;
        uniform highp sampler2D sphericalHarmonicsTextureB;

        uniform highp usampler2D sceneIndexesTexture;
        uniform vec2 sceneIndexesTextureSize;
        uniform int sceneCount;
    `;return t&&(r+=`
            uniform float sceneOpacity[${Ot.MaxScenes}];
            uniform int sceneVisibility[${Ot.MaxScenes}];
        `),e&&(r+=`
            uniform highp mat4 transforms[${Ot.MaxScenes}];
        `),r+=`
        ${s}
        uniform vec2 focal;
        uniform float orthoZoom;
        uniform int orthographicMode;
        uniform int pointCloudModeEnabled;
        uniform float inverseFocalAdjustment;
        uniform vec2 viewport;
        uniform vec2 basisViewport;
        uniform vec2 centersColorsTextureSize;
        uniform int sphericalHarmonicsDegree;
        uniform vec2 sphericalHarmonicsTextureSize;
        uniform int sphericalHarmonics8BitMode;
        uniform int sphericalHarmonicsMultiTextureMode;
        uniform float visibleRegionRadius;
        uniform float visibleRegionFadeStartRadius;
        uniform float firstRenderTime;
        uniform float currentTime;
        uniform int fadeInComplete;
        uniform vec3 sceneCenter;
        uniform float splatScale;
        uniform float sphericalHarmonics8BitCompressionRangeMin[${Ot.MaxScenes}];
        uniform float sphericalHarmonics8BitCompressionRangeMax[${Ot.MaxScenes}];

        varying vec4 vColor;
        varying vec2 vUv;
        varying vec2 vPosition;

        mat3 quaternionToRotationMatrix(float x, float y, float z, float w) {
            float s = 1.0 / sqrt(w * w + x * x + y * y + z * z);
        
            return mat3(
                1. - 2. * (y * y + z * z),
                2. * (x * y + w * z),
                2. * (x * z - w * y),
                2. * (x * y - w * z),
                1. - 2. * (x * x + z * z),
                2. * (y * z + w * x),
                2. * (x * z + w * y),
                2. * (y * z - w * x),
                1. - 2. * (x * x + y * y)
            );
        }

        const float sqrt8 = sqrt(8.0);
        const float minAlpha = 1.0 / 255.0;

        const vec4 encodeNorm4 = vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
        const uvec4 mask4 = uvec4(uint(0x000000FF), uint(0x0000FF00), uint(0x00FF0000), uint(0xFF000000));
        const uvec4 shift4 = uvec4(0, 8, 16, 24);
        vec4 uintToRGBAVec (uint u) {
           uvec4 urgba = mask4 & u;
           urgba = urgba >> shift4;
           vec4 rgba = vec4(urgba) * encodeNorm4;
           return rgba;
        }

        vec2 getDataUV(in int stride, in int offset, in vec2 dimensions) {
            vec2 samplerUV = vec2(0.0, 0.0);
            float d = float(splatIndex * uint(stride) + uint(offset)) / dimensions.x;
            samplerUV.y = float(floor(d)) / dimensions.y;
            samplerUV.x = fract(d);
            return samplerUV;
        }

        vec2 getDataUVF(in uint sIndex, in float stride, in uint offset, in vec2 dimensions) {
            vec2 samplerUV = vec2(0.0, 0.0);
            float d = float(uint(float(sIndex) * stride) + offset) / dimensions.x;
            samplerUV.y = float(floor(d)) / dimensions.y;
            samplerUV.x = fract(d);
            return samplerUV;
        }

        const float SH_C1 = 0.4886025119029199f;
        const float[5] SH_C2 = float[](1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);

        void main () {

            uint oddOffset = splatIndex & uint(0x00000001);
            uint doubleOddOffset = oddOffset * uint(2);
            bool isEven = oddOffset == uint(0);
            uint nearestEvenIndex = splatIndex - oddOffset;
            float fOddOffset = float(oddOffset);

            uvec4 sampledCenterColor = texture(centersColorsTexture, getDataUV(1, 0, centersColorsTextureSize));
            vec3 splatCenter = uintBitsToFloat(uvec3(sampledCenterColor.gba));

            uint sceneIndex = uint(0);
            if (sceneCount > 1) {
                sceneIndex = texture(sceneIndexesTexture, getDataUV(1, 0, sceneIndexesTextureSize)).r;
            }
            `,t&&(r+=`
                float splatOpacityFromScene = sceneOpacity[sceneIndex];
                int sceneVisible = sceneVisibility[sceneIndex];
                if (splatOpacityFromScene <= 0.01 || sceneVisible == 0) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }
            `),e?r+=`
                mat4 transform = transforms[sceneIndex];
                mat4 transformModelViewMatrix = viewMatrix * transform;
            `:r+="mat4 transformModelViewMatrix = modelViewMatrix;",r+=`
            float sh8BitCompressionRangeMinForScene = sphericalHarmonics8BitCompressionRangeMin[sceneIndex];
            float sh8BitCompressionRangeMaxForScene = sphericalHarmonics8BitCompressionRangeMax[sceneIndex];
            float sh8BitCompressionRangeForScene = sh8BitCompressionRangeMaxForScene - sh8BitCompressionRangeMinForScene;
            float sh8BitCompressionHalfRangeForScene = sh8BitCompressionRangeForScene / 2.0;
            vec3 vec8BitSHShift = vec3(sh8BitCompressionRangeMinForScene);

            vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0);

            vec4 clipCenter = projectionMatrix * viewCenter;

            float clip = 1.2 * clipCenter.w;
            if (clipCenter.z < -clip || clipCenter.x < -clip || clipCenter.x > clip || clipCenter.y < -clip || clipCenter.y > clip) {
                gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                return;
            }

            vec3 ndcCenter = clipCenter.xyz / clipCenter.w;

            vPosition = position.xy;
            vColor = uintToRGBAVec(sampledCenterColor.r);
        `,n>=1&&(r+=`   
            if (sphericalHarmonicsDegree >= 1) {
            `,e?r+=`
                    vec3 worldViewDir = normalize(splatCenter - vec3(inverse(transform) * vec4(cameraPosition, 1.0)));
                `:r+=`
                    vec3 worldViewDir = normalize(splatCenter - cameraPosition);
                `,r+=`
                vec3 sh1;
                vec3 sh2;
                vec3 sh3;
            `,n>=2&&(r+=`
                    vec3 sh4;
                    vec3 sh5;
                    vec3 sh6;
                    vec3 sh7;
                    vec3 sh8;
                `),n===1?r+=`
                    if (sphericalHarmonicsMultiTextureMode == 0) {
                        vec2 shUV = getDataUVF(nearestEvenIndex, 2.5, doubleOddOffset, sphericalHarmonicsTextureSize);
                        vec4 sampledSH0123 = texture(sphericalHarmonicsTexture, shUV);
                        shUV = getDataUVF(nearestEvenIndex, 2.5, doubleOddOffset + uint(1), sphericalHarmonicsTextureSize);
                        vec4 sampledSH4567 = texture(sphericalHarmonicsTexture, shUV);
                        shUV = getDataUVF(nearestEvenIndex, 2.5, doubleOddOffset + uint(2), sphericalHarmonicsTextureSize);
                        vec4 sampledSH891011 = texture(sphericalHarmonicsTexture, shUV);
                        sh1 = vec3(sampledSH0123.rgb) * (1.0 - fOddOffset) + vec3(sampledSH0123.ba, sampledSH4567.r) * fOddOffset;
                        sh2 = vec3(sampledSH0123.a, sampledSH4567.rg) * (1.0 - fOddOffset) + vec3(sampledSH4567.gba) * fOddOffset;
                        sh3 = vec3(sampledSH4567.ba, sampledSH891011.r) * (1.0 - fOddOffset) + vec3(sampledSH891011.rgb) * fOddOffset;
                    } else {
                        vec2 sampledSH01R = texture(sphericalHarmonicsTextureR, getDataUV(2, 0, sphericalHarmonicsTextureSize)).rg;
                        vec2 sampledSH23R = texture(sphericalHarmonicsTextureR, getDataUV(2, 1, sphericalHarmonicsTextureSize)).rg;
                        vec2 sampledSH01G = texture(sphericalHarmonicsTextureG, getDataUV(2, 0, sphericalHarmonicsTextureSize)).rg;
                        vec2 sampledSH23G = texture(sphericalHarmonicsTextureG, getDataUV(2, 1, sphericalHarmonicsTextureSize)).rg;
                        vec2 sampledSH01B = texture(sphericalHarmonicsTextureB, getDataUV(2, 0, sphericalHarmonicsTextureSize)).rg;
                        vec2 sampledSH23B = texture(sphericalHarmonicsTextureB, getDataUV(2, 1, sphericalHarmonicsTextureSize)).rg;
                        sh1 = vec3(sampledSH01R.rg, sampledSH23R.r);
                        sh2 = vec3(sampledSH01G.rg, sampledSH23G.r);
                        sh3 = vec3(sampledSH01B.rg, sampledSH23B.r);
                    }
                `:n===2&&(r+=`
                    vec4 sampledSH0123;
                    vec4 sampledSH4567;
                    vec4 sampledSH891011;

                    vec4 sampledSH0123R;
                    vec4 sampledSH0123G;
                    vec4 sampledSH0123B;

                    if (sphericalHarmonicsMultiTextureMode == 0) {
                        sampledSH0123 = texture(sphericalHarmonicsTexture, getDataUV(6, 0, sphericalHarmonicsTextureSize));
                        sampledSH4567 = texture(sphericalHarmonicsTexture, getDataUV(6, 1, sphericalHarmonicsTextureSize));
                        sampledSH891011 = texture(sphericalHarmonicsTexture, getDataUV(6, 2, sphericalHarmonicsTextureSize));
                        sh1 = sampledSH0123.rgb;
                        sh2 = vec3(sampledSH0123.a, sampledSH4567.rg);
                        sh3 = vec3(sampledSH4567.ba, sampledSH891011.r);
                    } else {
                        sampledSH0123R = texture(sphericalHarmonicsTextureR, getDataUV(2, 0, sphericalHarmonicsTextureSize));
                        sampledSH0123G = texture(sphericalHarmonicsTextureG, getDataUV(2, 0, sphericalHarmonicsTextureSize));
                        sampledSH0123B = texture(sphericalHarmonicsTextureB, getDataUV(2, 0, sphericalHarmonicsTextureSize));
                        sh1 = vec3(sampledSH0123R.rgb);
                        sh2 = vec3(sampledSH0123G.rgb);
                        sh3 = vec3(sampledSH0123B.rgb);
                    }
                `),r+=`
                    if (sphericalHarmonics8BitMode == 1) {
                        sh1 = sh1 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                        sh2 = sh2 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                        sh3 = sh3 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                    }
                    float x = worldViewDir.x;
                    float y = worldViewDir.y;
                    float z = worldViewDir.z;
                    vColor.rgb += SH_C1 * (-sh1 * y + sh2 * z - sh3 * x);
            `,n>=2&&(r+=`
                    if (sphericalHarmonicsDegree >= 2) {
                        float xx = x * x;
                        float yy = y * y;
                        float zz = z * z;
                        float xy = x * y;
                        float yz = y * z;
                        float xz = x * z;
                `,n===2&&(r+=`
                        if (sphericalHarmonicsMultiTextureMode == 0) {
                            vec4 sampledSH12131415 = texture(sphericalHarmonicsTexture, getDataUV(6, 3, sphericalHarmonicsTextureSize));
                            vec4 sampledSH16171819 = texture(sphericalHarmonicsTexture, getDataUV(6, 4, sphericalHarmonicsTextureSize));
                            vec4 sampledSH20212223 = texture(sphericalHarmonicsTexture, getDataUV(6, 5, sphericalHarmonicsTextureSize));
                            sh4 = sampledSH891011.gba;
                            sh5 = sampledSH12131415.rgb;
                            sh6 = vec3(sampledSH12131415.a, sampledSH16171819.rg);
                            sh7 = vec3(sampledSH16171819.ba, sampledSH20212223.r);
                            sh8 = sampledSH20212223.gba;
                        } else {
                            vec4 sampledSH4567R = texture(sphericalHarmonicsTextureR, getDataUV(2, 1, sphericalHarmonicsTextureSize));
                            vec4 sampledSH4567G = texture(sphericalHarmonicsTextureG, getDataUV(2, 1, sphericalHarmonicsTextureSize));
                            vec4 sampledSH4567B = texture(sphericalHarmonicsTextureB, getDataUV(2, 1, sphericalHarmonicsTextureSize));
                            sh4 = vec3(sampledSH0123R.a, sampledSH4567R.rg);
                            sh5 = vec3(sampledSH4567R.ba, sampledSH0123G.a);
                            sh6 = vec3(sampledSH4567G.rgb);
                            sh7 = vec3(sampledSH4567G.a, sampledSH0123B.a, sampledSH4567B.r);
                            sh8 = vec3(sampledSH4567B.gba);
                        }
                    `),r+=`
                        if (sphericalHarmonics8BitMode == 1) {
                            sh4 = sh4 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                            sh5 = sh5 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                            sh6 = sh6 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                            sh7 = sh7 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                            sh8 = sh8 * sh8BitCompressionRangeForScene + vec8BitSHShift;
                        }

                        vColor.rgb +=
                            (SH_C2[0] * xy) * sh4 +
                            (SH_C2[1] * yz) * sh5 +
                            (SH_C2[2] * (2.0 * zz - xx - yy)) * sh6 +
                            (SH_C2[3] * xz) * sh7 +
                            (SH_C2[4] * (xx - yy)) * sh8;
                    }
                `),r+=`

                vColor.rgb = clamp(vColor.rgb, vec3(0.), vec3(1.));

            }

            `),r}static getVertexShaderFadeIn(){return`
            if (fadeInComplete == 0) {
                float opacityAdjust = 1.0;
                float centerDist = length(splatCenter - sceneCenter);
                float renderTime = max(currentTime - firstRenderTime, 0.0);

                float fadeDistance = 0.75;
                float distanceLoadFadeInFactor = step(visibleRegionFadeStartRadius, centerDist);
                distanceLoadFadeInFactor = (1.0 - distanceLoadFadeInFactor) +
                                        (1.0 - clamp((centerDist - visibleRegionFadeStartRadius) / fadeDistance, 0.0, 1.0)) *
                                        distanceLoadFadeInFactor;
                opacityAdjust *= distanceLoadFadeInFactor;
                vColor.a *= opacityAdjust;
            }
        `}static getUniforms(e=!1,t=!1,n=0,s=1,r=!1){const o={sceneCenter:{type:"v3",value:new U},fadeInComplete:{type:"i",value:0},orthographicMode:{type:"i",value:0},visibleRegionFadeStartRadius:{type:"f",value:0},visibleRegionRadius:{type:"f",value:0},currentTime:{type:"f",value:0},firstRenderTime:{type:"f",value:0},centersColorsTexture:{type:"t",value:null},sphericalHarmonicsTexture:{type:"t",value:null},sphericalHarmonicsTextureR:{type:"t",value:null},sphericalHarmonicsTextureG:{type:"t",value:null},sphericalHarmonicsTextureB:{type:"t",value:null},sphericalHarmonics8BitCompressionRangeMin:{type:"f",value:[]},sphericalHarmonics8BitCompressionRangeMax:{type:"f",value:[]},focal:{type:"v2",value:new Ke},orthoZoom:{type:"f",value:1},inverseFocalAdjustment:{type:"f",value:1},viewport:{type:"v2",value:new Ke},basisViewport:{type:"v2",value:new Ke},debugColor:{type:"v3",value:new vt},centersColorsTextureSize:{type:"v2",value:new Ke(1024,1024)},sphericalHarmonicsDegree:{type:"i",value:n},sphericalHarmonicsTextureSize:{type:"v2",value:new Ke(1024,1024)},sphericalHarmonics8BitMode:{type:"i",value:0},sphericalHarmonicsMultiTextureMode:{type:"i",value:0},splatScale:{type:"f",value:s},pointCloudModeEnabled:{type:"i",value:r?1:0},sceneIndexesTexture:{type:"t",value:null},sceneIndexesTextureSize:{type:"v2",value:new Ke(1024,1024)},sceneCount:{type:"i",value:1}};for(let a=0;a<Ot.MaxScenes;a++)o.sphericalHarmonics8BitCompressionRangeMin.value.push(-3/2),o.sphericalHarmonics8BitCompressionRangeMax.value.push(Ot.SphericalHarmonics8BitCompressionRange/2);if(t){const a=[];for(let c=0;c<Ot.MaxScenes;c++)a.push(1);o.sceneOpacity={type:"f",value:a};const l=[];for(let c=0;c<Ot.MaxScenes;c++)l.push(1);o.sceneVisibility={type:"i",value:l}}if(e){const a=[];for(let l=0;l<Ot.MaxScenes;l++)a.push(new rt);o.transforms={type:"mat4",value:a}}return o}}class Mc{static build(e=!1,t=!1,n=!1,s=2048,r=1,o=!1,a=0,l=.3){let u=Lo.buildVertexShaderBase(e,t,a,`
            uniform vec2 covariancesTextureSize;
            uniform highp sampler2D covariancesTexture;
            uniform highp usampler2D covariancesTextureHalfFloat;
            uniform int covariancesAreHalfFloat;

            void fromCovarianceHalfFloatV4(uvec4 val, out vec4 first, out vec4 second) {
                vec2 r = unpackHalf2x16(val.r);
                vec2 g = unpackHalf2x16(val.g);
                vec2 b = unpackHalf2x16(val.b);

                first = vec4(r.x, r.y, g.x, g.y);
                second = vec4(b.x, b.y, 0.0, 0.0);
            }
        `);u+=Mc.buildVertexShaderProjection(n,t,s,l);const f=Mc.buildFragmentShader(),d=Lo.getUniforms(e,t,a,r,o);return d.covariancesTextureSize={type:"v2",value:new Ke(1024,1024)},d.covariancesTexture={type:"t",value:null},d.covariancesTextureHalfFloat={type:"t",value:null},d.covariancesAreHalfFloat={type:"i",value:0},new Kn({uniforms:d,vertexShader:u,fragmentShader:f,transparent:!0,alphaTest:1,blending:ir,depthTest:!0,depthWrite:!1,side:Ei})}static buildVertexShaderProjection(e,t,n,s){let r=`

            vec4 sampledCovarianceA;
            vec4 sampledCovarianceB;
            vec3 cov3D_M11_M12_M13;
            vec3 cov3D_M22_M23_M33;
            if (covariancesAreHalfFloat == 0) {
                sampledCovarianceA = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset,
                                                                            covariancesTextureSize));
                sampledCovarianceB = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset + uint(1),
                                                                            covariancesTextureSize));

                cov3D_M11_M12_M13 = vec3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) +
                                    vec3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) +
                                    vec3(sampledCovarianceB.gba) * fOddOffset;
            } else {
                uvec4 sampledCovarianceU = texture(covariancesTextureHalfFloat, getDataUV(1, 0, covariancesTextureSize));
                fromCovarianceHalfFloatV4(sampledCovarianceU, sampledCovarianceA, sampledCovarianceB);
                cov3D_M11_M12_M13 = sampledCovarianceA.rgb;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg);
            }
        
            // Construct the 3D covariance matrix
            mat3 Vrk = mat3(
                cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z,
                cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x, cov3D_M22_M23_M33.y,
                cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z
            );

            mat3 J;
            if (orthographicMode == 1) {
                // Since the projection is linear, we don't need an approximation
                J = transpose(mat3(orthoZoom, 0.0, 0.0,
                                0.0, orthoZoom, 0.0,
                                0.0, 0.0, 0.0));
            } else {
                // Construct the Jacobian of the affine approximation of the projection matrix. It will be used to transform the
                // 3D covariance matrix instead of using the actual projection matrix because that transformation would
                // require a non-linear component (perspective division) which would yield a non-gaussian result.
                float s = 1.0 / (viewCenter.z * viewCenter.z);
                J = mat3(
                    focal.x / viewCenter.z, 0., -(focal.x * viewCenter.x) * s,
                    0., focal.y / viewCenter.z, -(focal.y * viewCenter.y) * s,
                    0., 0., 0.
                );
            }

            // Concatenate the projection approximation with the model-view transformation
            mat3 W = transpose(mat3(transformModelViewMatrix));
            mat3 T = W * J;

            // Transform the 3D covariance matrix (Vrk) to compute the 2D covariance matrix
            mat3 cov2Dm = transpose(T) * Vrk * T;
            `;return e?r+=`
                float detOrig = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
                cov2Dm[0][0] += ${s};
                cov2Dm[1][1] += ${s};
                float detBlur = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
                vColor.a *= sqrt(max(detOrig / detBlur, 0.0));
                if (vColor.a < minAlpha) return;
            `:r+=`
                cov2Dm[0][0] += ${s};
                cov2Dm[1][1] += ${s};
            `,r+=`

            // We are interested in the upper-left 2x2 portion of the projected 3D covariance matrix because
            // we only care about the X and Y values. We want the X-diagonal, cov2Dm[0][0],
            // the Y-diagonal, cov2Dm[1][1], and the correlation between the two cov2Dm[0][1]. We don't
            // need cov2Dm[1][0] because it is a symetric matrix.
            vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

            // We now need to solve for the eigen-values and eigen vectors of the 2D covariance matrix
            // so that we can determine the 2D basis for the splat. This is done using the method described
            // here: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
            // After calculating the eigen-values and eigen-vectors, we calculate the basis for rendering the splat
            // by normalizing the eigen-vectors and then multiplying them by (sqrt(8) * sqrt(eigen-value)), which is
            // equal to scaling them by sqrt(8) standard deviations.
            //
            // This is a different approach than in the original work at INRIA. In that work they compute the
            // max extents of the projected splat in screen space to form a screen-space aligned bounding rectangle
            // which forms the geometry that is actually rasterized. The dimensions of that bounding box are 3.0
            // times the square root of the maximum eigen-value, or 3 standard deviations. They then use the inverse
            // 2D covariance matrix (called 'conic') in the CUDA rendering thread to determine fragment opacity by
            // calculating the full gaussian: exp(-0.5 * (X - mean) * conic * (X - mean)) * splat opacity
            float a = cov2Dv.x;
            float d = cov2Dv.z;
            float b = cov2Dv.y;
            float D = a * d - b * b;
            float trace = a + d;
            float traceOver2 = 0.5 * trace;
            float term2 = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
            float eigenValue1 = traceOver2 + term2;
            float eigenValue2 = traceOver2 - term2;

            if (pointCloudModeEnabled == 1) {
                eigenValue1 = eigenValue2 = 0.2;
            }

            if (eigenValue2 <= 0.0) return;

            vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
            // since the eigen vectors are orthogonal, we derive the second one from the first
            vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

            // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
            vec2 basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), ${parseInt(n)}.0);
            vec2 basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), ${parseInt(n)}.0);
            `,t&&(r+=`
                vColor.a *= splatOpacityFromScene;
            `),r+=`
            vec2 ndcOffset = vec2(vPosition.x * basisVector1 + vPosition.y * basisVector2) *
                             basisViewport * 2.0 * inverseFocalAdjustment;

            vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
            gl_Position = quadPos;

            // Scale the position data we send to the fragment shader
            vPosition *= sqrt8;
        `,r+=Lo.getVertexShaderFadeIn(),r+="}",r}static buildFragmentShader(){let e=`
            precision highp float;
            #include <common>
 
            uniform vec3 debugColor;

            varying vec4 vColor;
            varying vec2 vUv;
            varying vec2 vPosition;
        `;return e+=`
            void main () {
                // Compute the positional squared distance from the center of the splat to the current fragment.
                float A = dot(vPosition, vPosition);
                // Since the positional data in vPosition has been scaled by sqrt(8), the squared result will be
                // scaled by a factor of 8. If the squared result is larger than 8, it means it is outside the ellipse
                // defined by the rectangle formed by vPosition. It also means it's farther
                // away than sqrt(8) standard deviations from the mean.
                if (A > 8.0) discard;
                vec3 color = vColor.rgb;

                // Since the rendered splat is scaled by sqrt(8), the inverse covariance matrix that is part of
                // the gaussian formula becomes the identity matrix. We're then left with (X - mean) * (X - mean),
                // and since 'mean' is zero, we have X * X, which is the same as A:
                float opacity = exp(-0.5 * A) * vColor.a;

                gl_FragColor = vec4(color.rgb, opacity);
            }
        `,e}}class Cc{static build(e=!1,t=!1,n=1,s=!1,r=0){let a=Lo.buildVertexShaderBase(e,t,r,`
            uniform vec2 scaleRotationsTextureSize;
            uniform highp sampler2D scaleRotationsTexture;
            varying mat3 vT;
            varying vec2 vQuadCenter;
            varying vec2 vFragCoord;
        `);a+=Cc.buildVertexShaderProjection();const l=Cc.buildFragmentShader(),c=Lo.getUniforms(e,t,r,n,s);return c.scaleRotationsTexture={type:"t",value:null},c.scaleRotationsTextureSize={type:"v2",value:new Ke(1024,1024)},new Kn({uniforms:c,vertexShader:a,fragmentShader:l,transparent:!0,alphaTest:1,blending:ir,depthTest:!0,depthWrite:!1,side:Ei})}static buildVertexShaderProjection(){let e=`

            vec4 scaleRotationA = texture(scaleRotationsTexture, getDataUVF(nearestEvenIndex, 1.5,
                                                                            oddOffset, scaleRotationsTextureSize));
            vec4 scaleRotationB = texture(scaleRotationsTexture, getDataUVF(nearestEvenIndex, 1.5,
                                                                            oddOffset + uint(1), scaleRotationsTextureSize));

            vec3 scaleRotation123 = vec3(scaleRotationA.rgb) * (1.0 - fOddOffset) +
                                    vec3(scaleRotationA.ba, scaleRotationB.r) * fOddOffset;
            vec3 scaleRotation456 = vec3(scaleRotationA.a, scaleRotationB.rg) * (1.0 - fOddOffset) +
                                    vec3(scaleRotationB.gba) * fOddOffset;

            float missingW = sqrt(1.0 - scaleRotation456.x * scaleRotation456.x - scaleRotation456.y *
                                    scaleRotation456.y - scaleRotation456.z * scaleRotation456.z);
            mat3 R = quaternionToRotationMatrix(scaleRotation456.r, scaleRotation456.g, scaleRotation456.b, missingW);
            mat3 S = mat3(scaleRotation123.r, 0.0, 0.0,
                            0.0, scaleRotation123.g, 0.0,
                            0.0, 0.0, scaleRotation123.b);
            
            mat3 L = R * S;

            mat3x4 splat2World = mat3x4(vec4(L[0], 0.0),
                                        vec4(L[1], 0.0),
                                        vec4(splatCenter.x, splatCenter.y, splatCenter.z, 1.0));

            mat4 world2ndc = transpose(projectionMatrix * transformModelViewMatrix);

            mat3x4 ndc2pix = mat3x4(vec4(viewport.x / 2.0, 0.0, 0.0, (viewport.x - 1.0) / 2.0),
                                    vec4(0.0, viewport.y / 2.0, 0.0, (viewport.y - 1.0) / 2.0),
                                    vec4(0.0, 0.0, 0.0, 1.0));

            mat3 T = transpose(splat2World) * world2ndc * ndc2pix;
            vec3 normal = vec3(viewMatrix * vec4(L[0][2], L[1][2], L[2][2], 0.0));
        `;return e+=`

                mat4 splat2World4 = mat4(vec4(L[0], 0.0),
                                        vec4(L[1], 0.0),
                                        vec4(L[2], 0.0),
                                        vec4(splatCenter.x, splatCenter.y, splatCenter.z, 1.0));

                mat4 Tt = transpose(transpose(splat2World4) * world2ndc);

                vec4 tempPoint1 = Tt * vec4(1.0, 0.0, 0.0, 1.0);
                tempPoint1 /= tempPoint1.w;

                vec4 tempPoint2 = Tt * vec4(0.0, 1.0, 0.0, 1.0);
                tempPoint2 /= tempPoint2.w;

                vec4 center = Tt * vec4(0.0, 0.0, 0.0, 1.0);
                center /= center.w;

                vec2 basisVector1 = tempPoint1.xy - center.xy;
                vec2 basisVector2 = tempPoint2.xy - center.xy;

                vec2 basisVector1Screen = basisVector1 * 0.5 * viewport;
                vec2 basisVector2Screen = basisVector2 * 0.5 * viewport;

                const float minPix = 1.;
                if (length(basisVector1Screen) < minPix || length(basisVector2Screen) < minPix) {
                    
            vec3 T0 = vec3(T[0][0], T[0][1], T[0][2]);
            vec3 T1 = vec3(T[1][0], T[1][1], T[1][2]);
            vec3 T3 = vec3(T[2][0], T[2][1], T[2][2]);

            vec3 tempPoint = vec3(1.0, 1.0, -1.0);
            float distance = (T3.x * T3.x * tempPoint.x) + (T3.y * T3.y * tempPoint.y) + (T3.z * T3.z * tempPoint.z);
            vec3 f = (1.0 / distance) * tempPoint;
            if (abs(distance) < 0.00001) return;

            float pointImageX = (T0.x * T3.x * f.x) + (T0.y * T3.y * f.y) + (T0.z * T3.z * f.z);
            float pointImageY = (T1.x * T3.x * f.x) + (T1.y * T3.y * f.y) + (T1.z * T3.z * f.z);
            vec2 pointImage = vec2(pointImageX, pointImageY);

            float tempX = (T0.x * T0.x * f.x) + (T0.y * T0.y * f.y) + (T0.z * T0.z * f.z);
            float tempY = (T1.x * T1.x * f.x) + (T1.y * T1.y * f.y) + (T1.z * T1.z * f.z);
            vec2 temp = vec2(tempX, tempY);

            vec2 halfExtend = pointImage * pointImage - temp;
            vec2 extent = sqrt(max(vec2(0.0001), halfExtend));
            float radius = max(extent.x, extent.y);

            vec2 ndcOffset = ((position.xy * radius * 3.0) * basisViewport * 2.0);

            vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
            gl_Position = quadPos;

            vT = T;
            vQuadCenter = pointImage;
            vFragCoord = (quadPos.xy * 0.5 + 0.5) * viewport;
        
                } else {
                    vec2 ndcOffset = vec2(position.x * basisVector1 + position.y * basisVector2) * 3.0 * inverseFocalAdjustment;
                    vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
                    gl_Position = quadPos;

                    vT = T;
                    vQuadCenter = center.xy;
                    vFragCoord = (quadPos.xy * 0.5 + 0.5) * viewport;
                }
            `,e+=Lo.getVertexShaderFadeIn(),e+="}",e}static buildFragmentShader(){return`
            precision highp float;
            #include <common>

            uniform vec3 debugColor;

            varying vec4 vColor;
            varying vec2 vUv;
            varying vec2 vPosition;
            varying mat3 vT;
            varying vec2 vQuadCenter;
            varying vec2 vFragCoord;

            void main () {

                const float FilterInvSquare = 2.0;
                const float near_n = 0.2;
                const float T = 1.0;

                vec2 xy = vQuadCenter;
                vec3 Tu = vT[0];
                vec3 Tv = vT[1];
                vec3 Tw = vT[2];
                vec3 k = vFragCoord.x * Tw - Tu;
                vec3 l = vFragCoord.y * Tw - Tv;
                vec3 p = cross(k, l);
                if (p.z == 0.0) discard;
                vec2 s = vec2(p.x / p.z, p.y / p.z);
                float rho3d = (s.x * s.x + s.y * s.y); 
                vec2 d = vec2(xy.x - vFragCoord.x, xy.y - vFragCoord.y);
                float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

                // compute intersection and depth
                float rho = min(rho3d, rho2d);
                float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
                if (depth < near_n) discard;
                //  vec4 nor_o = collected_normal_opacity[j];
                //  float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
                float opa = vColor.a;

                float power = -0.5f * rho;
                if (power > 0.0f) discard;

                // Eq. (2) from 3D Gaussian splatting paper.
                // Obtain alpha by multiplying with Gaussian opacity
                // and its exponential falloff from mean.
                // Avoid numerical instabilities (see paper appendix). 
                float alpha = min(0.99f, opa * exp(power));
                if (alpha < 1.0f / 255.0f) discard;
                float test_T = T * (1.0 - alpha);
                if (test_T < 0.0001)discard;

                float w = alpha * T;
                gl_FragColor = vec4(vColor.rgb, w);
            }
        `}}class TE{static build(e){const t=new jn;t.setIndex([0,1,2,0,2,3]);const n=new Float32Array(12),s=new Li(n,3);t.setAttribute("position",s),s.setXYZ(0,-1,-1,0),s.setXYZ(1,-1,1,0),s.setXYZ(2,1,1,0),s.setXYZ(3,1,-1,0),s.needsUpdate=!0;const r=new Iy().copy(t),o=new Uint32Array(e),a=new Sy(o,1,!1);return a.setUsage(wS),r.setAttribute("splatIndex",a),r.instanceCount=0,r}}class EE extends dn{constructor(e,t=new U,n=new Gt,s=new U(1,1,1),r=1,o=1,a=!0){super(),this.splatBuffer=e,this.position.copy(t),this.quaternion.copy(n),this.scale.copy(s),this.transform=new rt,this.minimumAlpha=r,this.opacity=o,this.visible=a}copyTransformData(e){this.position.copy(e.position),this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.transform.copy(e.transform)}updateTransform(e){e?(this.matrixWorldAutoUpdate&&this.updateWorldMatrix(!0,!1),this.transform.copy(this.matrixWorld)):(this.matrixAutoUpdate&&this.updateMatrix(),this.transform.copy(this.matrix))}}class yh{static idGen=0;constructor(e,t,n,s){this.min=new U().copy(e),this.max=new U().copy(t),this.boundingBox=new es(this.min,this.max),this.center=new U().copy(this.max).sub(this.min).multiplyScalar(.5).add(this.min),this.depth=n,this.children=[],this.data=null,this.id=s||yh.idGen++}}class Ba{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.sceneDimensions=new U,this.sceneMin=new U,this.sceneMax=new U,this.rootNode=null,this.nodesWithIndexes=[],this.splatMesh=null}static convertWorkerSubTreeNode(e){const t=new U().fromArray(e.min),n=new U().fromArray(e.max),s=new yh(t,n,e.depth,e.id);if(e.data.indexes){s.data={indexes:[]};for(let r of e.data.indexes)s.data.indexes.push(r)}if(e.children)for(let r of e.children)s.children.push(Ba.convertWorkerSubTreeNode(r));return s}static convertWorkerSubTree(e,t){const n=new Ba(e.maxDepth,e.maxCentersPerNode);n.sceneMin=new U().fromArray(e.sceneMin),n.sceneMax=new U().fromArray(e.sceneMax),n.splatMesh=t,n.rootNode=Ba.convertWorkerSubTreeNode(e.rootNode);const s=(r,o)=>{r.children.length===0&&o(r);for(let a of r.children)s(a,o)};return n.nodesWithIndexes=[],s(n.rootNode,r=>{r.data&&r.data.indexes&&r.data.indexes.length>0&&n.nodesWithIndexes.push(r)}),n}}function wE(i){let e=0;class t{constructor(l,c){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]]}containsPoint(l){return l[0]>=this.min[0]&&l[0]<=this.max[0]&&l[1]>=this.min[1]&&l[1]<=this.max[1]&&l[2]>=this.min[2]&&l[2]<=this.max[2]}}class n{constructor(l,c){this.maxDepth=l,this.maxCentersPerNode=c,this.sceneDimensions=[],this.sceneMin=[],this.sceneMax=[],this.rootNode=null,this.addedIndexes={},this.nodesWithIndexes=[],this.splatMesh=null,this.disposed=!1}}class s{constructor(l,c,u,f){this.min=[l[0],l[1],l[2]],this.max=[c[0],c[1],c[2]],this.center=[(c[0]-l[0])*.5+l[0],(c[1]-l[1])*.5+l[1],(c[2]-l[2])*.5+l[2]],this.depth=u,this.children=[],this.data=null,this.id=f||e++}}processSplatTreeNode=function(a,l,c,u){const f=l.data.indexes.length;if(f<a.maxCentersPerNode||l.depth>a.maxDepth){const _=[];for(let v=0;v<l.data.indexes.length;v++)a.addedIndexes[l.data.indexes[v]]||(_.push(l.data.indexes[v]),a.addedIndexes[l.data.indexes[v]]=!0);l.data.indexes=_,l.data.indexes.sort((v,A)=>v>A?1:-1),a.nodesWithIndexes.push(l);return}const d=[l.max[0]-l.min[0],l.max[1]-l.min[1],l.max[2]-l.min[2]],h=[d[0]*.5,d[1]*.5,d[2]*.5],x=[l.min[0]+h[0],l.min[1]+h[1],l.min[2]+h[2]],p=[new t([x[0]-h[0],x[1],x[2]-h[2]],[x[0],x[1]+h[1],x[2]]),new t([x[0],x[1],x[2]-h[2]],[x[0]+h[0],x[1]+h[1],x[2]]),new t([x[0],x[1],x[2]],[x[0]+h[0],x[1]+h[1],x[2]+h[2]]),new t([x[0]-h[0],x[1],x[2]],[x[0],x[1]+h[1],x[2]+h[2]]),new t([x[0]-h[0],x[1]-h[1],x[2]-h[2]],[x[0],x[1],x[2]]),new t([x[0],x[1]-h[1],x[2]-h[2]],[x[0]+h[0],x[1],x[2]]),new t([x[0],x[1]-h[1],x[2]],[x[0]+h[0],x[1],x[2]+h[2]]),new t([x[0]-h[0],x[1]-h[1],x[2]],[x[0],x[1],x[2]+h[2]])],g=[];for(let _=0;_<p.length;_++)g[_]=[];const m=[0,0,0];for(let _=0;_<f;_++){const v=l.data.indexes[_],A=c[v];m[0]=u[A],m[1]=u[A+1],m[2]=u[A+2];for(let S=0;S<p.length;S++)p[S].containsPoint(m)&&g[S].push(v)}for(let _=0;_<p.length;_++){const v=new s(p[_].min,p[_].max,l.depth+1);v.data={indexes:g[_]},l.children.push(v)}l.data={};for(let _ of l.children)processSplatTreeNode(a,_,c,u)};const r=(a,l,c)=>{const u=[0,0,0],f=[0,0,0],d=[],h=Math.floor(a.length/4);for(let p=0;p<h;p++){const g=p*4,m=a[g],_=a[g+1],v=a[g+2],A=Math.round(a[g+3]);(p===0||m<u[0])&&(u[0]=m),(p===0||m>f[0])&&(f[0]=m),(p===0||_<u[1])&&(u[1]=_),(p===0||_>f[1])&&(f[1]=_),(p===0||v<u[2])&&(u[2]=v),(p===0||v>f[2])&&(f[2]=v),d.push(A)}const x=new n(l,c);return x.sceneMin=u,x.sceneMax=f,x.rootNode=new s(x.sceneMin,x.sceneMax,0),x.rootNode.data={indexes:d},x};function o(a,l,c){const u=[];for(let d of a){const h=Math.floor(d.length/4);for(let x=0;x<h;x++){const p=x*4,g=Math.round(d[p+3]);u[g]=p}}const f=[];for(let d of a){const h=r(d,l,c);f.push(h),processSplatTreeNode(h,h.rootNode,u,d)}i.postMessage({subTrees:f})}i.onmessage=a=>{a.data.process&&o(a.data.process.centers,a.data.process.maxDepth,a.data.process.maxCentersPerNode)}}function RE(i,e,t,n,s){i.postMessage({process:{centers:e,maxDepth:n,maxCentersPerNode:s}},t)}function IE(){return new Worker(URL.createObjectURL(new Blob(["(",wE.toString(),")(self)"],{type:"application/javascript"})))}class DE{constructor(e,t){this.maxDepth=e,this.maxCentersPerNode=t,this.subTrees=[],this.splatMesh=null}dispose(){this.diposeSplatTreeWorker(),this.disposed=!0}diposeSplatTreeWorker(){this.splatTreeWorker&&this.splatTreeWorker.terminate(),this.splatTreeWorker=null}processSplatMesh=function(e,t=()=>!0,n,s){this.splatTreeWorker||(this.splatTreeWorker=IE()),this.splatMesh=e,this.subTrees=[];const r=new U,o=(a,l)=>{const c=new Float32Array(l*4);let u=0;for(let f=0;f<l;f++){const d=f+a;if(t(d)){e.getSplatCenter(d,r);const h=u*4;c[h]=r.x,c[h+1]=r.y,c[h+2]=r.z,c[h+3]=d,u++}}return c};return new Promise(a=>{const l=()=>this.disposed?(this.diposeSplatTreeWorker(),a(),!0):!1;n&&n(!1),pi(()=>{if(l())return;const c=[];if(e.dynamicMode){let u=0;for(let f=0;f<e.scenes.length;f++){const h=e.getScene(f).splatBuffer.getSplatCount(),x=o(u,h);c.push(x),u+=h}}else{const u=o(0,e.getSplatCount());c.push(u)}this.splatTreeWorker.onmessage=u=>{l()||u.data.subTrees&&(s&&s(!1),pi(()=>{if(!l()){for(let f of u.data.subTrees){const d=Ba.convertWorkerSubTree(f,e);this.subTrees.push(d)}this.diposeSplatTreeWorker(),s&&s(!0),pi(()=>{a()})}}))},pi(()=>{if(l())return;n&&n(!0);const u=c.map(f=>f.buffer);RE(this.splatTreeWorker,c,u,this.maxDepth,this.maxCentersPerNode)})})})};countLeaves(){let e=0;return this.visitLeaves(()=>{e++}),e}visitLeaves(e){const t=(n,s)=>{n.children.length===0&&s(n);for(let r of n.children)t(r,s)};for(let n of this.subTrees)t(n.rootNode,e)}}function PE(i){const e={};function t(n){if(e[n]!==void 0)return e[n];let s;switch(n){case"WEBGL_depth_texture":s=i.getExtension("WEBGL_depth_texture")||i.getExtension("MOZ_WEBGL_depth_texture")||i.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":s=i.getExtension("EXT_texture_filter_anisotropic")||i.getExtension("MOZ_EXT_texture_filter_anisotropic")||i.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":s=i.getExtension("WEBGL_compressed_texture_s3tc")||i.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":s=i.getExtension("WEBGL_compressed_texture_pvrtc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:s=i.getExtension(n)}return e[n]=s,s}return{has:function(n){return t(n)!==null},init:function(n){n.isWebGL2?(t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance")):(t("WEBGL_depth_texture"),t("OES_texture_float"),t("OES_texture_half_float"),t("OES_texture_half_float_linear"),t("OES_standard_derivatives"),t("OES_element_index_uint"),t("OES_vertex_array_object"),t("ANGLE_instanced_arrays")),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture")},get:function(n){const s=t(n);return s===null&&console.warn("THREE.WebGLRenderer: "+n+" extension not supported."),s}}}function FE(i,e,t){let n;function s(){if(n!==void 0)return n;if(e.has("EXT_texture_filter_anisotropic")===!0){const M=e.get("EXT_texture_filter_anisotropic");n=i.getParameter(M.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else n=0;return n}function r(M){if(M==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";M="mediump"}return M==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}const o=typeof WebGL2RenderingContext<"u"&&i.constructor.name==="WebGL2RenderingContext";let a=t.precision!==void 0?t.precision:"highp";const l=r(a);l!==a&&(console.warn("THREE.WebGLRenderer:",a,"not supported, using",l,"instead."),a=l);const c=o||e.has("WEBGL_draw_buffers"),u=t.logarithmicDepthBuffer===!0,f=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),d=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),h=i.getParameter(i.MAX_TEXTURE_SIZE),x=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),p=i.getParameter(i.MAX_VERTEX_ATTRIBS),g=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),m=i.getParameter(i.MAX_VARYING_VECTORS),_=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),v=d>0,A=o||e.has("OES_texture_float"),S=v&&A,y=o?i.getParameter(i.MAX_SAMPLES):0;return{isWebGL2:o,drawBuffers:c,getMaxAnisotropy:s,getMaxPrecision:r,precision:a,logarithmicDepthBuffer:u,maxTextures:f,maxVertexTextures:d,maxTextureSize:h,maxCubemapSize:x,maxAttributes:p,maxVertexUniforms:g,maxVaryings:m,maxFragmentUniforms:_,vertexTextures:v,floatFragmentTextures:A,floatVertexTextures:S,maxSamples:y}}const Ua={Default:0,Instant:2},Bo={None:0,Info:3},qm=new jn,LE=new Yr,$l=6,BE=4,UE=4,OE=4,NE=6,zE=8,ef=4,tf=4,Ym=1,kE=.012,HE=.003,Qm=1,Km=16777216;class Tn extends un{constructor(e=Cs.ThreeD,t=!1,n=!1,s=!1,r=1,o=!0,a=!1,l=!1,c=1024,u=Bo.None,f=0,d=1,h=.3){super(qm,LE),this.renderer=void 0,this.splatRenderMode=e,this.dynamicMode=t,this.enableOptionalEffects=n,this.halfPrecisionCovariancesOnGPU=s,this.devicePixelRatio=r,this.enableDistancesComputationOnGPU=o,this.integerBasedDistancesComputation=a,this.antialiased=l,this.kernel2DSize=h,this.maxScreenSpaceSplatSize=c,this.logLevel=u,this.sphericalHarmonicsDegree=f,this.minSphericalHarmonicsDegree=0,this.sceneFadeInRateMultiplier=d,this.scenes=[],this.splatTree=null,this.baseSplatTree=null,this.splatDataTextures={},this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new es,this.calculatedSceneCenter=new U,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!1,this.lastRenderer=null,this.visible=!1}static buildScenes(e,t,n){const s=[];s.length=t.length;for(let r=0;r<t.length;r++){const o=t[r],a=n[r]||{};let l=a.position||[0,0,0],c=a.rotation||[0,0,0,1],u=a.scale||[1,1,1];const f=new U().fromArray(l),d=new Gt().fromArray(c),h=new U().fromArray(u),x=Tn.createScene(o,f,d,h,a.splatAlphaRemovalThreshold||1,a.opacity,a.visible);e.add(x),s[r]=x}return s}static createScene(e,t,n,s,r,o=1,a=!0){return new EE(e,t,n,s,r,o,a)}static buildSplatIndexMaps(e){const t=[],n=[];let s=0;for(let r=0;r<e.length;r++){const a=e[r].getMaxSplatCount();for(let l=0;l<a;l++)t[s]=l,n[s]=r,s++}return{localSplatIndexMap:t,sceneIndexMap:n}}buildSplatTree=function(e=[],t,n){return new Promise(s=>{this.disposeSplatTree(),this.baseSplatTree=new DE(8,1e3);const r=performance.now(),o=new Zt;this.baseSplatTree.processSplatMesh(this,a=>{this.getSplatColor(a,o);const l=this.getSceneIndexForSplat(a),c=e[l]||1;return o.w>=c},t,n).then(()=>{const a=performance.now()-r;if(this.logLevel>=Bo.Info&&console.log("SplatTree build: "+a+" ms"),this.disposed)s();else{this.splatTree=this.baseSplatTree,this.baseSplatTree=null;let l=0,c=0,u=0;this.splatTree.visitLeaves(f=>{const d=f.data.indexes.length;d>0&&(c+=d,u++,l++)}),this.logLevel>=Bo.Info&&(console.log(`SplatTree leaves: ${this.splatTree.countLeaves()}`),console.log(`SplatTree leaves with splats:${l}`),c=c/u,console.log(`Avg splat count per node: ${c}`),console.log(`Total splat count: ${this.getSplatCount()}`)),s()}})})};build(e,t,n=!0,s=!1,r,o,a=!0){this.sceneOptions=t,this.finalBuild=s;const l=Tn.getTotalMaxSplatCountForSplatBuffers(e),c=Tn.buildScenes(this,e,t);if(n)for(let p=0;p<this.scenes.length&&p<c.length;p++){const g=c[p],m=this.getScene(p);g.copyTransformData(m)}this.scenes=c;let u=3;for(let p of e){const g=p.getMinSphericalHarmonicsDegree();g<u&&(u=g)}this.minSphericalHarmonicsDegree=Math.min(u,this.sphericalHarmonicsDegree);let f=!1;if(e.length!==this.lastBuildScenes.length)f=!0;else for(let p=0;p<e.length;p++)if(e[p]!==this.lastBuildScenes[p].splatBuffer){f=!0;break}let d=!0;if((this.scenes.length!==1||this.lastBuildSceneCount!==this.scenes.length||this.lastBuildMaxSplatCount!==l||f)&&(d=!1),!d){this.boundingBox=new es,a||(this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.firstRenderTime=-1),this.lastBuildScenes=[],this.lastBuildSplatCount=0,this.lastBuildMaxSplatCount=0,this.disposeMeshData(),this.geometry=TE.build(l),this.splatRenderMode===Cs.ThreeD?this.material=Mc.build(this.dynamicMode,this.enableOptionalEffects,this.antialiased,this.maxScreenSpaceSplatSize,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree,this.kernel2DSize):this.material=Cc.build(this.dynamicMode,this.enableOptionalEffects,this.splatScale,this.pointCloudModeEnabled,this.minSphericalHarmonicsDegree);const p=Tn.buildSplatIndexMaps(e);this.globalSplatIndexToLocalSplatIndexMap=p.localSplatIndexMap,this.globalSplatIndexToSceneIndexMap=p.sceneIndexMap}const h=this.getSplatCount(!0);this.enableDistancesComputationOnGPU&&this.setupDistancesComputationTransformFeedback();const x=this.refreshGPUDataFromSplatBuffers(d);for(let p=0;p<this.scenes.length;p++)this.lastBuildScenes[p]=this.scenes[p];return this.lastBuildSplatCount=h,this.lastBuildMaxSplatCount=this.getMaxSplatCount(),this.lastBuildSceneCount=this.scenes.length,s&&this.scenes.length>0&&this.buildSplatTree(t.map(p=>p.splatAlphaRemovalThreshold||1),r,o).then(()=>{this.onSplatTreeReadyCallback&&this.onSplatTreeReadyCallback(this.splatTree),this.onSplatTreeReadyCallback=null}),this.visible=this.scenes.length>0,x}freeIntermediateSplatData(){const e=t=>{delete t.source.data,delete t.image,t.onUpdate=null};delete this.splatDataTextures.baseData.covariances,delete this.splatDataTextures.baseData.centers,delete this.splatDataTextures.baseData.colors,delete this.splatDataTextures.baseData.sphericalHarmonics,delete this.splatDataTextures.centerColors.data,delete this.splatDataTextures.covariances.data,this.splatDataTextures.sphericalHarmonics&&delete this.splatDataTextures.sphericalHarmonics.data,this.splatDataTextures.sceneIndexes&&delete this.splatDataTextures.sceneIndexes.data,this.splatDataTextures.centerColors.texture.needsUpdate=!0,this.splatDataTextures.centerColors.texture.onUpdate=()=>{e(this.splatDataTextures.centerColors.texture)},this.splatDataTextures.covariances.texture.needsUpdate=!0,this.splatDataTextures.covariances.texture.onUpdate=()=>{e(this.splatDataTextures.covariances.texture)},this.splatDataTextures.sphericalHarmonics&&(this.splatDataTextures.sphericalHarmonics.texture?(this.splatDataTextures.sphericalHarmonics.texture.needsUpdate=!0,this.splatDataTextures.sphericalHarmonics.texture.onUpdate=()=>{e(this.splatDataTextures.sphericalHarmonics.texture)}):this.splatDataTextures.sphericalHarmonics.textures.forEach(t=>{t.needsUpdate=!0,t.onUpdate=()=>{e(t)}})),this.splatDataTextures.sceneIndexes&&(this.splatDataTextures.sceneIndexes.texture.needsUpdate=!0,this.splatDataTextures.sceneIndexes.texture.onUpdate=()=>{e(this.splatDataTextures.sceneIndexes.texture)})}dispose(){this.disposeMeshData(),this.disposeTextures(),this.disposeSplatTree(),this.enableDistancesComputationOnGPU&&(this.computeDistancesOnGPUSyncTimeout&&(clearTimeout(this.computeDistancesOnGPUSyncTimeout),this.computeDistancesOnGPUSyncTimeout=null),this.disposeDistancesComputationGPUResources()),this.scenes=[],this.distancesTransformFeedback={id:null,vertexShader:null,fragmentShader:null,program:null,centersBuffer:null,sceneIndexesBuffer:null,outDistancesBuffer:null,centersLoc:-1,modelViewProjLoc:-1,sceneIndexesLoc:-1,transformsLocs:[]},this.renderer=null,this.globalSplatIndexToLocalSplatIndexMap=[],this.globalSplatIndexToSceneIndexMap=[],this.lastBuildSplatCount=0,this.lastBuildScenes=[],this.lastBuildMaxSplatCount=0,this.lastBuildSceneCount=0,this.firstRenderTime=-1,this.finalBuild=!1,this.webGLUtils=null,this.boundingBox=new es,this.calculatedSceneCenter=new U,this.maxSplatDistanceFromSceneCenter=0,this.visibleRegionBufferRadius=0,this.visibleRegionRadius=0,this.visibleRegionFadeStartRadius=0,this.visibleRegionChanging=!1,this.splatScale=1,this.pointCloudModeEnabled=!1,this.disposed=!0,this.lastRenderer=null,this.visible=!1}disposeMeshData(){this.geometry&&this.geometry!==qm&&(this.geometry.dispose(),this.geometry=null),this.material&&(this.material.dispose(),this.material=null)}disposeTextures(){for(let e in this.splatDataTextures)if(this.splatDataTextures.hasOwnProperty(e)){const t=this.splatDataTextures[e];t.texture&&(t.texture.dispose(),t.texture=null)}this.splatDataTextures=null}disposeSplatTree(){this.splatTree&&(this.splatTree.dispose(),this.splatTree=null),this.baseSplatTree&&(this.baseSplatTree.dispose(),this.baseSplatTree=null)}getSplatTree(){return this.splatTree}onSplatTreeReady(e){this.onSplatTreeReadyCallback=e}getDataForDistancesComputation(e,t){const n=this.integerBasedDistancesComputation?this.getIntegerCenters(e,t,!0):this.getFloatCenters(e,t,!0),s=this.getSceneIndexes(e,t);return{centers:n,sceneIndexes:s}}refreshGPUDataFromSplatBuffers(e){const t=this.getSplatCount(!0);this.refreshDataTexturesFromSplatBuffers(e);const n=e?this.lastBuildSplatCount:0,{centers:s,sceneIndexes:r}=this.getDataForDistancesComputation(n,t-1);return this.enableDistancesComputationOnGPU&&this.refreshGPUBuffersForDistancesComputation(s,r,e),{from:n,to:t-1,count:t-n,centers:s,sceneIndexes:r}}refreshGPUBuffersForDistancesComputation(e,t,n=!1){const s=n?this.lastBuildSplatCount:0;this.updateGPUCentersBufferForDistancesComputation(n,e,s),this.updateGPUTransformIndexesBufferForDistancesComputation(n,t,s)}refreshDataTexturesFromSplatBuffers(e){const t=this.getSplatCount(!0),n=this.lastBuildSplatCount,s=t-1;e?this.updateBaseDataFromSplatBuffers(n,s):(this.setupDataTextures(),this.updateBaseDataFromSplatBuffers()),this.updateDataTexturesFromBaseData(n,s),this.updateVisibleRegion(e)}setupDataTextures(){const e=this.getMaxSplatCount(),t=this.getSplatCount(!0);this.disposeTextures();const n=(M,E)=>{const b=new Ke(4096,1024);for(;b.x*b.y*M<e*E;)b.y*=2;return b},s=M=>M>=1?NE:UE,r=M=>{const E=s(M),b=n(E,6);return{elementsPerTexelStored:E,texSize:b}};let o=this.getTargetCovarianceCompressionLevel();const a=0,l=this.getTargetSphericalHarmonicsCompressionLevel();let c,u,f;if(this.splatRenderMode===Cs.ThreeD){const M=r(o);M.texSize.x*M.texSize.y>Km&&o===0&&(o=1),c=new Float32Array(e*$l)}else u=new Float32Array(e*3),f=new Float32Array(e*4);const d=new Float32Array(e*3),h=new Uint8Array(e*4);let x=Float32Array;l===1?x=Uint16Array:l===2&&(x=Uint8Array);const p=Fo(this.minSphericalHarmonicsDegree),g=this.minSphericalHarmonicsDegree?new x(e*p):void 0,m=n(tf,4),_=new Uint32Array(m.x*m.y*tf);Tn.updateCenterColorsPaddedData(0,t-1,d,h,_);const v=new Ss(_,m.x,m.y,Do,Ii);if(v.internalFormat="RGBA32UI",v.needsUpdate=!0,this.material.uniforms.centersColorsTexture.value=v,this.material.uniforms.centersColorsTextureSize.value.copy(m),this.material.uniformsNeedUpdate=!0,this.splatDataTextures={baseData:{covariances:c,scales:u,rotations:f,centers:d,colors:h,sphericalHarmonics:g},centerColors:{data:_,texture:v,size:m}},this.splatRenderMode===Cs.ThreeD){const M=r(o),E=M.elementsPerTexelStored,b=M.texSize;let C=o>=1?Uint32Array:Float32Array;const P=o>=1?zE:OE,B=new C(b.x*b.y*P);o===0?B.set(c):Tn.updatePaddedCompressedCovariancesTextureData(c,B,0,0,c.length);let N;if(o>=1)N=new Ss(B,b.x,b.y,Do,Ii),N.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=N;else{N=new Ss(B,b.x,b.y,Yn,ki),this.material.uniforms.covariancesTexture.value=N;const k=new Ss(new Uint32Array(32),2,2,Do,Ii);k.internalFormat="RGBA32UI",this.material.uniforms.covariancesTextureHalfFloat.value=k,k.needsUpdate=!0}N.needsUpdate=!0,this.material.uniforms.covariancesAreHalfFloat.value=o>=1?1:0,this.material.uniforms.covariancesTextureSize.value.copy(b),this.splatDataTextures.covariances={data:B,texture:N,size:b,compressionLevel:o,elementsPerTexelStored:E,elementsPerTexelAllocated:P}}else{const E=n(ef,6);let b=Float32Array,C=ki;const P=new b(E.x*E.y*ef);Tn.updateScaleRotationsPaddedData(0,t-1,u,f,P);const B=new Ss(P,E.x,E.y,Yn,C);B.needsUpdate=!0,this.material.uniforms.scaleRotationsTexture.value=B,this.material.uniforms.scaleRotationsTextureSize.value.copy(E),this.splatDataTextures.scaleRotations={data:P,texture:B,size:E,compressionLevel:a}}if(g){const M=l===2?ls:Qr;let E=p;E%2!==0&&E++;const b=4,C=Yn;let P=n(b,E);if(P.x*P.y<=Km){const B=P.x*P.y*b,N=new x(B);for(let V=0;V<t;V++){const F=p*V,q=E*V;for(let G=0;G<p;G++)N[q+G]=g[F+G]}const k=new Ss(N,P.x,P.y,C,M);k.needsUpdate=!0,this.material.uniforms.sphericalHarmonicsTexture.value=k,this.splatDataTextures.sphericalHarmonics={componentCount:p,paddedComponentCount:E,data:N,textureCount:1,texture:k,size:P,compressionLevel:l,elementsPerTexel:b}}else{const B=p/3;E=B,E%2!==0&&E++,P=n(b,E);const N=P.x*P.y*b,k=[this.material.uniforms.sphericalHarmonicsTextureR,this.material.uniforms.sphericalHarmonicsTextureG,this.material.uniforms.sphericalHarmonicsTextureB],V=[],F=[];for(let q=0;q<3;q++){const G=new x(N);V.push(G);for(let ce=0;ce<t;ce++){const ye=p*ce,Ie=E*ce;if(B>=3){for(let Ne=0;Ne<3;Ne++)G[Ie+Ne]=g[ye+q*3+Ne];if(B>=8)for(let Ne=0;Ne<5;Ne++)G[Ie+3+Ne]=g[ye+9+q*5+Ne]}}const j=new Ss(G,P.x,P.y,C,M);F.push(j),j.needsUpdate=!0,k[q].value=j}this.material.uniforms.sphericalHarmonicsMultiTextureMode.value=1,this.splatDataTextures.sphericalHarmonics={componentCount:p,componentCountPerChannel:B,paddedComponentCount:E,data:V,textureCount:3,textures:F,size:P,compressionLevel:l,elementsPerTexel:b}}this.material.uniforms.sphericalHarmonicsTextureSize.value.copy(P),this.material.uniforms.sphericalHarmonics8BitMode.value=l===2?1:0;for(let B=0;B<this.scenes.length;B++){const N=this.scenes[B].splatBuffer;this.material.uniforms.sphericalHarmonics8BitCompressionRangeMin.value[B]=N.minSphericalHarmonicsCoeff,this.material.uniforms.sphericalHarmonics8BitCompressionRangeMax.value[B]=N.maxSphericalHarmonicsCoeff}this.material.uniformsNeedUpdate=!0}const A=n(Ym,4),S=new Uint32Array(A.x*A.y*Ym);for(let M=0;M<t;M++)S[M]=this.globalSplatIndexToSceneIndexMap[M];const y=new Ss(S,A.x,A.y,Wc,Ii);y.internalFormat="R32UI",y.needsUpdate=!0,this.material.uniforms.sceneIndexesTexture.value=y,this.material.uniforms.sceneIndexesTextureSize.value.copy(A),this.material.uniformsNeedUpdate=!0,this.splatDataTextures.sceneIndexes={data:S,texture:y,size:A},this.material.uniforms.sceneCount.value=this.scenes.length}updateBaseDataFromSplatBuffers(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0;this.fillSplatDataArrays(this.splatDataTextures.baseData.covariances,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,this.splatDataTextures.baseData.sphericalHarmonics,void 0,s,o,l,e,t,e)}updateDataTexturesFromBaseData(e,t){const n=this.splatDataTextures.covariances,s=n?n.compressionLevel:void 0,r=this.splatDataTextures.scaleRotations,o=r?r.compressionLevel:void 0,a=this.splatDataTextures.sphericalHarmonics,l=a?a.compressionLevel:0,c=this.splatDataTextures.centerColors,u=c.data,f=c.texture;Tn.updateCenterColorsPaddedData(e,t,this.splatDataTextures.baseData.centers,this.splatDataTextures.baseData.colors,u);const d=this.renderer?this.renderer.properties.get(f):null;if(!d||!d.__webglTexture?f.needsUpdate=!0:this.updateDataTexture(u,c.texture,c.size,d,tf,BE,4,e,t),n){const _=n.texture,v=e*$l,A=t*$l;if(s===0)for(let y=v;y<=A;y++){const M=this.splatDataTextures.baseData.covariances[y];n.data[y]=M}else Tn.updatePaddedCompressedCovariancesTextureData(this.splatDataTextures.baseData.covariances,n.data,e*n.elementsPerTexelAllocated,v,A);const S=this.renderer?this.renderer.properties.get(_):null;!S||!S.__webglTexture?_.needsUpdate=!0:s===0?this.updateDataTexture(n.data,n.texture,n.size,S,n.elementsPerTexelStored,$l,4,e,t):this.updateDataTexture(n.data,n.texture,n.size,S,n.elementsPerTexelAllocated,n.elementsPerTexelAllocated,2,e,t)}if(r){const _=r.data,v=r.texture,A=6,S=o===0?4:2;Tn.updateScaleRotationsPaddedData(e,t,this.splatDataTextures.baseData.scales,this.splatDataTextures.baseData.rotations,_);const y=this.renderer?this.renderer.properties.get(v):null;!y||!y.__webglTexture?v.needsUpdate=!0:this.updateDataTexture(_,r.texture,r.size,y,ef,A,S,e,t)}const h=this.splatDataTextures.baseData.sphericalHarmonics;if(h){let _=4;l===1?_=2:l===2&&(_=1);const v=(y,M,E,b,C)=>{const P=this.renderer?this.renderer.properties.get(y):null;!P||!P.__webglTexture?y.needsUpdate=!0:this.updateDataTexture(b,y,M,P,E,C,_,e,t)},A=a.componentCount,S=a.paddedComponentCount;if(a.textureCount===1){const y=a.data;for(let M=e;M<=t;M++){const E=A*M,b=S*M;for(let C=0;C<A;C++)y[b+C]=h[E+C]}v(a.texture,a.size,a.elementsPerTexel,y,S)}else{const y=a.componentCountPerChannel;for(let M=0;M<3;M++){const E=a.data[M];for(let b=e;b<=t;b++){const C=A*b,P=S*b;if(y>=3){for(let B=0;B<3;B++)E[P+B]=h[C+M*3+B];if(y>=8)for(let B=0;B<5;B++)E[P+3+B]=h[C+9+M*5+B]}}v(a.textures[M],a.size,a.elementsPerTexel,E,S)}}}const x=this.splatDataTextures.sceneIndexes,p=x.data;for(let _=this.lastBuildSplatCount;_<=t;_++)p[_]=this.globalSplatIndexToSceneIndexMap[_];const g=x.texture,m=this.renderer?this.renderer.properties.get(g):null;!m||!m.__webglTexture?g.needsUpdate=!0:this.updateDataTexture(p,x.texture,x.size,m,1,1,1,this.lastBuildSplatCount,t)}getTargetCovarianceCompressionLevel(){return this.halfPrecisionCovariancesOnGPU?1:0}getTargetSphericalHarmonicsCompressionLevel(){return Math.max(1,this.getMaximumSplatBufferCompressionLevel())}getMaximumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel>e)&&(e=s.compressionLevel)}return e}getMinimumSplatBufferCompressionLevel(){let e;for(let t=0;t<this.scenes.length;t++){const s=this.getScene(t).splatBuffer;(t===0||s.compressionLevel<e)&&(e=s.compressionLevel)}return e}static computeTextureUpdateRegion(e,t,n,s,r){const o=r/s,a=e*o,l=Math.floor(a/n),c=l*n*s,u=t*o,f=Math.floor(u/n),d=f*n*s+n*s;return{dataStart:c,dataEnd:d,startRow:l,endRow:f}}updateDataTexture(e,t,n,s,r,o,a,l,c){const u=this.renderer.getContext(),f=Tn.computeTextureUpdateRegion(l,c,n.x,r,o),d=f.dataEnd-f.dataStart,h=new e.constructor(e.buffer,f.dataStart*a,d),x=f.endRow-f.startRow+1,p=this.webGLUtils.convert(t.type),g=this.webGLUtils.convert(t.format,t.colorSpace),m=u.getParameter(u.TEXTURE_BINDING_2D);u.bindTexture(u.TEXTURE_2D,s.__webglTexture),u.texSubImage2D(u.TEXTURE_2D,0,0,f.startRow,n.x,x,g,p,h),u.bindTexture(u.TEXTURE_2D,m)}static updatePaddedCompressedCovariancesTextureData(e,t,n,s,r){let o=new DataView(t.buffer),a=n,l=0;for(let c=s;c<=r;c+=2)o.setUint16(a*2,e[c],!0),o.setUint16(a*2+2,e[c+1],!0),a+=2,l++,l>=3&&(a+=2,l=0)}static updateCenterColorsPaddedData(e,t,n,s,r){for(let o=e;o<=t;o++){const a=o*4,l=o*3,c=o*4;r[c]=_1(s,a),r[c+1]=Yu(n[l]),r[c+2]=Yu(n[l+1]),r[c+3]=Yu(n[l+2])}}static updateScaleRotationsPaddedData(e,t,n,s,r){for(let a=e;a<=t;a++){const l=a*3,c=a*4,u=a*6;r[u]=n[l],r[u+1]=n[l+1],r[u+2]=n[l+2],r[u+3]=s[c],r[u+4]=s[c+1],r[u+5]=s[c+2]}}updateVisibleRegion(e){const t=this.getSplatCount(!0),n=new U;if(!e){const r=new U;this.scenes.forEach(o=>{r.add(o.splatBuffer.sceneCenter)}),r.multiplyScalar(1/this.scenes.length),this.calculatedSceneCenter.copy(r),this.material.uniforms.sceneCenter.value.copy(this.calculatedSceneCenter),this.material.uniformsNeedUpdate=!0}const s=e?this.lastBuildSplatCount:0;for(let r=s;r<t;r++){this.getSplatCenter(r,n,!0);const o=n.sub(this.calculatedSceneCenter).length();o>this.maxSplatDistanceFromSceneCenter&&(this.maxSplatDistanceFromSceneCenter=o)}this.maxSplatDistanceFromSceneCenter-this.visibleRegionBufferRadius>Qm&&(this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter,this.visibleRegionRadius=Math.max(this.visibleRegionBufferRadius-Qm,0)),this.finalBuild&&(this.visibleRegionRadius=this.visibleRegionBufferRadius=this.maxSplatDistanceFromSceneCenter),this.updateVisibleRegionFadeDistance()}updateVisibleRegionFadeDistance(e=Ua.Default){const t=kE*this.sceneFadeInRateMultiplier,n=HE*this.sceneFadeInRateMultiplier,s=this.finalBuild?t:n,r=e===Ua.Default?s:n;this.visibleRegionFadeStartRadius=(this.visibleRegionRadius-this.visibleRegionFadeStartRadius)*r+this.visibleRegionFadeStartRadius;const a=(this.visibleRegionBufferRadius>0?this.visibleRegionFadeStartRadius/this.visibleRegionBufferRadius:0)>.99,l=a||e===Ua.Instant?1:0;this.material.uniforms.visibleRegionFadeStartRadius.value=this.visibleRegionFadeStartRadius,this.material.uniforms.visibleRegionRadius.value=this.visibleRegionRadius,this.material.uniforms.firstRenderTime.value=this.firstRenderTime,this.material.uniforms.currentTime.value=performance.now(),this.material.uniforms.fadeInComplete.value=l,this.material.uniformsNeedUpdate=!0,this.visibleRegionChanging=!a}updateRenderIndexes(e,t){const n=this.geometry;n.attributes.splatIndex.set(e),n.attributes.splatIndex.needsUpdate=!0,t>0&&this.firstRenderTime===-1&&(this.firstRenderTime=performance.now()),n.instanceCount=t,n.setDrawRange(0,t)}updateTransforms(){for(let e=0;e<this.scenes.length;e++)this.getScene(e).updateTransform(this.dynamicMode)}updateUniforms=(function(){const e=new Ke;return function(t,n,s,r,o,a){if(this.getSplatCount()>0){if(e.set(t.x*this.devicePixelRatio,t.y*this.devicePixelRatio),this.material.uniforms.viewport.value.copy(e),this.material.uniforms.basisViewport.value.set(1/e.x,1/e.y),this.material.uniforms.focal.value.set(n,s),this.material.uniforms.orthographicMode.value=r?1:0,this.material.uniforms.orthoZoom.value=o,this.material.uniforms.inverseFocalAdjustment.value=a,this.dynamicMode)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.transforms.value[c].copy(this.getScene(c).transform);if(this.enableOptionalEffects)for(let c=0;c<this.scenes.length;c++)this.material.uniforms.sceneOpacity.value[c]=jt(this.getScene(c).opacity,0,1),this.material.uniforms.sceneVisibility.value[c]=this.getScene(c).visible?1:0,this.material.uniformsNeedUpdate=!0;this.material.uniformsNeedUpdate=!0}}})();setSplatScale(e=1){this.splatScale=e,this.material.uniforms.splatScale.value=e,this.material.uniformsNeedUpdate=!0}getSplatScale(){return this.splatScale}setPointCloudModeEnabled(e){this.pointCloudModeEnabled=e,this.material.uniforms.pointCloudModeEnabled.value=e?1:0,this.material.uniformsNeedUpdate=!0}getPointCloudModeEnabled(){return this.pointCloudModeEnabled}getSplatDataTextures(){return this.splatDataTextures}getSplatCount(e=!1){return e?Tn.getTotalSplatCountForScenes(this.scenes):this.lastBuildSplatCount}static getTotalSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getSplatCount());return t}static getTotalSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getSplatCount();return t}getMaxSplatCount(){return Tn.getTotalMaxSplatCountForScenes(this.scenes)}static getTotalMaxSplatCountForScenes(e){let t=0;for(let n of e)n&&n.splatBuffer&&(t+=n.splatBuffer.getMaxSplatCount());return t}static getTotalMaxSplatCountForSplatBuffers(e){let t=0;for(let n of e)t+=n.getMaxSplatCount();return t}disposeDistancesComputationGPUResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.vao&&(e.deleteVertexArray(this.distancesTransformFeedback.vao),this.distancesTransformFeedback.vao=null),this.distancesTransformFeedback.program&&(e.deleteProgram(this.distancesTransformFeedback.program),e.deleteShader(this.distancesTransformFeedback.vertexShader),e.deleteShader(this.distancesTransformFeedback.fragmentShader),this.distancesTransformFeedback.program=null,this.distancesTransformFeedback.vertexShader=null,this.distancesTransformFeedback.fragmentShader=null),this.disposeDistancesComputationGPUBufferResources(),this.distancesTransformFeedback.id&&(e.deleteTransformFeedback(this.distancesTransformFeedback.id),this.distancesTransformFeedback.id=null)}disposeDistancesComputationGPUBufferResources(){if(!this.renderer)return;const e=this.renderer.getContext();this.distancesTransformFeedback.centersBuffer&&(this.distancesTransformFeedback.centersBuffer=null,e.deleteBuffer(this.distancesTransformFeedback.centersBuffer)),this.distancesTransformFeedback.outDistancesBuffer&&(e.deleteBuffer(this.distancesTransformFeedback.outDistancesBuffer),this.distancesTransformFeedback.outDistancesBuffer=null)}setRenderer(e){if(e!==this.renderer){this.renderer=e;const t=this.renderer.getContext(),n=new PE(t),s=new FE(t,n,{});if(n.init(s),this.webGLUtils=new tx(t,n),this.enableDistancesComputationOnGPU&&this.getSplatCount()>0){this.setupDistancesComputationTransformFeedback();const{centers:r,sceneIndexes:o}=this.getDataForDistancesComputation(0,this.getSplatCount()-1);this.refreshGPUBuffersForDistancesComputation(r,o)}}}setupDistancesComputationTransformFeedback=(function(){let e;return function(){const t=this.getMaxSplatCount();if(!this.renderer)return;const n=this.lastRenderer!==this.renderer,s=e!==t;if(!n&&!s)return;n?this.disposeDistancesComputationGPUResources():s&&this.disposeDistancesComputationGPUBufferResources();const r=this.renderer.getContext(),o=(d,h,x)=>{const p=d.createShader(h);if(!p)return console.error("Fatal error: gl could not create a shader object."),null;if(d.shaderSource(p,x),d.compileShader(p),!d.getShaderParameter(p,d.COMPILE_STATUS)){let m="unknown";h===d.VERTEX_SHADER?m="vertex shader":h===d.FRAGMENT_SHADER&&(m="fragement shader");const _=d.getShaderInfoLog(p);return console.error("Failed to compile "+m+" with these errors:"+_),d.deleteShader(p),null}return p};let a;this.integerBasedDistancesComputation?(a=`#version 300 es
                in ivec4 center;
                flat out int distance;`,this.dynamicMode?a+=`
                        in uint sceneIndex;
                        uniform ivec4 transforms[${Ot.MaxScenes}];
                        void main(void) {
                            ivec4 transform = transforms[sceneIndex];
                            distance = center.x * transform.x + center.y * transform.y + center.z * transform.z + transform.w * center.w;
                        }
                    `:a+=`
                        uniform ivec3 modelViewProj;
                        void main(void) {
                            distance = center.x * modelViewProj.x + center.y * modelViewProj.y + center.z * modelViewProj.z;
                        }
                    `):(a=`#version 300 es
                in vec4 center;
                flat out float distance;`,this.dynamicMode?a+=`
                        in uint sceneIndex;
                        uniform mat4 transforms[${Ot.MaxScenes}];
                        void main(void) {
                            vec4 transformedCenter = transforms[sceneIndex] * vec4(center.xyz, 1.0);
                            distance = transformedCenter.z;
                        }
                    `:a+=`
                        uniform vec3 modelViewProj;
                        void main(void) {
                            distance = center.x * modelViewProj.x + center.y * modelViewProj.y + center.z * modelViewProj.z;
                        }
                    `);const l=`#version 300 es
                precision lowp float;
                out vec4 fragColor;
                void main(){}
            `,c=r.getParameter(r.VERTEX_ARRAY_BINDING),u=r.getParameter(r.CURRENT_PROGRAM),f=u?r.getProgramParameter(u,r.DELETE_STATUS):!1;if(n&&(this.distancesTransformFeedback.vao=r.createVertexArray()),r.bindVertexArray(this.distancesTransformFeedback.vao),n){const d=r.createProgram(),h=o(r,r.VERTEX_SHADER,a),x=o(r,r.FRAGMENT_SHADER,l);if(!h||!x)throw new Error("Could not compile shaders for distances computation on GPU.");if(r.attachShader(d,h),r.attachShader(d,x),r.transformFeedbackVaryings(d,["distance"],r.SEPARATE_ATTRIBS),r.linkProgram(d),!r.getProgramParameter(d,r.LINK_STATUS)){const g=r.getProgramInfoLog(d);throw console.error("Fatal error: Failed to link program: "+g),r.deleteProgram(d),r.deleteShader(x),r.deleteShader(h),new Error("Could not link shaders for distances computation on GPU.")}this.distancesTransformFeedback.program=d,this.distancesTransformFeedback.vertexShader=h,this.distancesTransformFeedback.vertexShader=x}if(r.useProgram(this.distancesTransformFeedback.program),this.distancesTransformFeedback.centersLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"center"),this.dynamicMode){this.distancesTransformFeedback.sceneIndexesLoc=r.getAttribLocation(this.distancesTransformFeedback.program,"sceneIndex");for(let d=0;d<this.scenes.length;d++)this.distancesTransformFeedback.transformsLocs[d]=r.getUniformLocation(this.distancesTransformFeedback.program,`transforms[${d}]`)}else this.distancesTransformFeedback.modelViewProjLoc=r.getUniformLocation(this.distancesTransformFeedback.program,"modelViewProj");(n||s)&&(this.distancesTransformFeedback.centersBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?r.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,r.INT,0,0):r.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,r.FLOAT,!1,0,0),this.dynamicMode&&(this.distancesTransformFeedback.sceneIndexesBuffer=r.createBuffer(),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),r.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),r.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,r.UNSIGNED_INT,0,0))),(n||s)&&(this.distancesTransformFeedback.outDistancesBuffer=r.createBuffer()),r.bindBuffer(r.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),r.bufferData(r.ARRAY_BUFFER,t*4,r.STATIC_READ),n&&(this.distancesTransformFeedback.id=r.createTransformFeedback()),r.bindTransformFeedback(r.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),r.bindBufferBase(r.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),u&&f!==!0&&r.useProgram(u),c&&r.bindVertexArray(c),this.lastRenderer=this.renderer,e=t}})();updateGPUCentersBufferForDistancesComputation(e,t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=this.integerBasedDistancesComputation?Uint32Array:Float32Array,a=16,l=n*a;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,l,t);else{const c=new o(this.getMaxSplatCount()*a);c.set(t),s.bufferData(s.ARRAY_BUFFER,c,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}updateGPUTransformIndexesBufferForDistancesComputation(e,t,n){if(!this.renderer||!this.dynamicMode)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao);const o=n*4;if(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),e)s.bufferSubData(s.ARRAY_BUFFER,o,t);else{const a=new Uint32Array(this.getMaxSplatCount()*4);a.set(t),s.bufferData(s.ARRAY_BUFFER,a,s.STATIC_DRAW)}s.bindBuffer(s.ARRAY_BUFFER,null),r&&s.bindVertexArray(r)}getSceneIndexes(e,t){let n;const s=t-e+1;n=new Uint32Array(s);for(let r=e;r<=t;r++)n[r]=this.globalSplatIndexToSceneIndexMap[r];return n}fillTransformsArray=(function(){const e=[];return function(t){e.length!==t.length&&(e.length=t.length);for(let n=0;n<this.scenes.length;n++){const r=this.getScene(n).transform.elements;for(let o=0;o<16;o++)e[n*16+o]=r[o]}t.set(e)}})();computeDistancesOnGPU=(function(){const e=new rt;return function(t,n){if(!this.renderer)return;const s=this.renderer.getContext(),r=s.getParameter(s.VERTEX_ARRAY_BINDING),o=s.getParameter(s.CURRENT_PROGRAM),a=o?s.getProgramParameter(o,s.DELETE_STATUS):!1;if(s.bindVertexArray(this.distancesTransformFeedback.vao),s.useProgram(this.distancesTransformFeedback.program),s.enable(s.RASTERIZER_DISCARD),this.dynamicMode)for(let u=0;u<this.scenes.length;u++)if(e.copy(this.getScene(u).transform),e.premultiply(t),this.integerBasedDistancesComputation){const f=Tn.getIntegerMatrixArray(e),d=[f[2],f[6],f[10],f[14]];s.uniform4i(this.distancesTransformFeedback.transformsLocs[u],d[0],d[1],d[2],d[3])}else s.uniformMatrix4fv(this.distancesTransformFeedback.transformsLocs[u],!1,e.elements);else if(this.integerBasedDistancesComputation){const u=Tn.getIntegerMatrixArray(t),f=[u[2],u[6],u[10]];s.uniform3i(this.distancesTransformFeedback.modelViewProjLoc,f[0],f[1],f[2])}else{const u=[t.elements[2],t.elements[6],t.elements[10]];s.uniform3f(this.distancesTransformFeedback.modelViewProjLoc,u[0],u[1],u[2])}s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.centersBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc),this.integerBasedDistancesComputation?s.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc,4,s.INT,0,0):s.vertexAttribPointer(this.distancesTransformFeedback.centersLoc,4,s.FLOAT,!1,0,0),this.dynamicMode&&(s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.sceneIndexesBuffer),s.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc),s.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc,1,s.UNSIGNED_INT,0,0)),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,this.distancesTransformFeedback.id),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,this.distancesTransformFeedback.outDistancesBuffer),s.beginTransformFeedback(s.POINTS),s.drawArrays(s.POINTS,0,this.getSplatCount()),s.endTransformFeedback(),s.bindBufferBase(s.TRANSFORM_FEEDBACK_BUFFER,0,null),s.bindTransformFeedback(s.TRANSFORM_FEEDBACK,null),s.disable(s.RASTERIZER_DISCARD);const l=s.fenceSync(s.SYNC_GPU_COMMANDS_COMPLETE,0);s.flush();const c=new Promise(u=>{const f=()=>{if(this.disposed)u();else switch(s.clientWaitSync(l,0,0)){case s.TIMEOUT_EXPIRED:return this.computeDistancesOnGPUSyncTimeout=setTimeout(f),this.computeDistancesOnGPUSyncTimeout;case s.WAIT_FAILED:throw new Error("should never get here");default:this.computeDistancesOnGPUSyncTimeout=null,s.deleteSync(l);const p=s.getParameter(s.VERTEX_ARRAY_BINDING);s.bindVertexArray(this.distancesTransformFeedback.vao),s.bindBuffer(s.ARRAY_BUFFER,this.distancesTransformFeedback.outDistancesBuffer),s.getBufferSubData(s.ARRAY_BUFFER,0,n),s.bindBuffer(s.ARRAY_BUFFER,null),p&&s.bindVertexArray(p),u()}};this.computeDistancesOnGPUSyncTimeout=setTimeout(f)});return o&&a!==!0&&s.useProgram(o),r&&s.bindVertexArray(r),c}})();getLocalSplatParameters(e,t,n){n==null&&(n=!this.dynamicMode),t.splatBuffer=this.getSplatBufferForSplat(e),t.localIndex=this.getSplatLocalIndex(e),t.sceneTransform=n?this.getSceneTransformForSplat(e):null}fillSplatDataArrays(e,t,n,s,r,o,a,l=0,c=0,u=1,f,d,h=0,x){const p=new U;p.x=void 0,p.y=void 0,this.splatRenderMode===Cs.ThreeD?p.z=void 0:p.z=1;const g=new rt;let m=0,_=this.scenes.length-1;x!=null&&x>=0&&x<=this.scenes.length&&(m=x,_=x);for(let v=m;v<=_;v++){a==null&&(a=!this.dynamicMode);const A=this.getScene(v),S=A.splatBuffer;let y;if(a&&(this.getSceneTransform(v,g),y=g),e&&S.fillSplatCovarianceArray(e,y,f,d,h,l),t||n){if(!t||!n)throw new Error('SplatMesh::fillSplatDataArrays() -> "scales" and "rotations" must both be valid.');S.fillSplatScaleRotationArray(t,n,y,f,d,h,c,p)}s&&S.fillSplatCenterArray(s,y,f,d,h),r&&S.fillSplatColorArray(r,A.minimumAlpha,f,d,h),o&&S.fillSphericalHarmonicsArray(o,this.minSphericalHarmonicsDegree,y,f,d,h,u),h+=S.getSplatCount()}}getIntegerCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e);let o,a=n?4:3;o=new Int32Array(s*a);for(let l=0;l<s;l++){for(let c=0;c<3;c++)o[l*a+c]=Math.round(r[l*3+c]*1e3);n&&(o[l*a+3]=1e3)}return o}getFloatCenters(e,t,n=!1){const s=t-e+1,r=new Float32Array(s*3);if(this.fillSplatDataArrays(null,null,null,r,null,null,void 0,void 0,void 0,void 0,e),!n)return r;let o=new Float32Array(s*4);for(let a=0;a<s;a++){for(let l=0;l<3;l++)o[a*4+l]=r[a*3+l];o[a*4+3]=1}return o}getSplatCenter=(function(){const e={};return function(t,n,s){this.getLocalSplatParameters(t,e,s),e.splatBuffer.getSplatCenter(e.localIndex,n,e.sceneTransform)}})();getSplatScaleAndRotation=(function(){const e={},t=new U;return function(n,s,r,o){this.getLocalSplatParameters(n,e,o),t.x=void 0,t.y=void 0,t.z=void 0,this.splatRenderMode===Cs.TwoD&&(t.z=0),e.splatBuffer.getSplatScaleAndRotation(e.localIndex,s,r,e.sceneTransform,t)}})();getSplatColor=(function(){const e={};return function(t,n){this.getLocalSplatParameters(t,e),e.splatBuffer.getSplatColor(e.localIndex,n)}})();getSceneTransform(e,t){const n=this.getScene(e);n.updateTransform(this.dynamicMode),t.copy(n.transform)}getScene(e){if(e<0||e>=this.scenes.length)throw new Error("SplatMesh::getScene() -> Invalid scene index.");return this.scenes[e]}getSceneCount(){return this.scenes.length}getSplatBufferForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).splatBuffer}getSceneIndexForSplat(e){return this.globalSplatIndexToSceneIndexMap[e]}getSceneTransformForSplat(e){return this.getScene(this.globalSplatIndexToSceneIndexMap[e]).transform}getSplatLocalIndex(e){return this.globalSplatIndexToLocalSplatIndexMap[e]}static getIntegerMatrixArray(e){const t=e.elements,n=[];for(let s=0;s<16;s++)n[s]=Math.round(t[s]*1e3);return n}computeBoundingBox(e=!1,t){let n=this.getSplatCount();if(t!=null){if(t<0||t>=this.scenes.length)throw new Error("SplatMesh::computeBoundingBox() -> Invalid scene index.");n=this.scenes[t].splatBuffer.getSplatCount()}const s=new Float32Array(n*3);this.fillSplatDataArrays(null,null,null,s,null,null,e,void 0,void 0,void 0,void 0,t);const r=new U,o=new U;for(let a=0;a<n;a++){const l=a*3,c=s[l],u=s[l+1],f=s[l+2];(a===0||c<r.x)&&(r.x=c),(a===0||u<r.y)&&(r.y=u),(a===0||f<r.z)&&(r.z=f),(a===0||c>o.x)&&(o.x=c),(a===0||u>o.y)&&(o.y=u),(a===0||f>o.z)&&(o.z=f)}return new es(r,o)}}var VE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEbA2AAAGAQf39/f39/f39/f39/f39/fwBgAAF/AhIBA2VudgZtZW1vcnkCAwCAgAQDBAMAAQIHVAQRX193YXNtX2NhbGxfY3RvcnMAABhfX3dhc21fYXBwbHlfZGF0YV9yZWxvY3MAAAtzb3J0SW5kZXhlcwABE2Vtc2NyaXB0ZW5fdGxzX2luaXQAAgqWEAMDAAELihAEAXwDewN/A30gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEBA0AgAyABQQJ0IgVqIAIgACAFaigCAEECdGooAgAiBTYCACAFIAogBSAKSBshCiAFIA0gBSANShshDSABQQFqIgEgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiFWooAgAiFkECdGooAgAiFEcEQAJ/IAX9CQI4IAggFEEGdGoiDv0JAgwgDioCHP0gASAOKgIs/SACIA4qAjz9IAP95gEgBf0JAiggDv0JAgggDioCGP0gASAOKgIo/SACIA4qAjj9IAP95gEgBf0JAgggDv0JAgAgDioCEP0gASAOKgIg/SACIA4qAjD9IAP95gEgBf0JAhggDv0JAgQgDioCFP0gASAOKgIk/SACIA4qAjT9IAP95gH95AH95AH95AEiEf1f/QwAAAAAAECPQAAAAAAAQI9AIhL98gEiE/0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBP9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/REgDv0cAQJ/IBEgEf0NCAkKCwwNDg8AAAAAAAAAAP1fIBL98gEiEf0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9HAICfyAR/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAyESIBQhDwsgAyAVaiABIBZBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmogEf0bA2oiDjYCACAOIAogCiAOShshCiAOIA0gDSAOSBshDSACQQFqIgIgC0cNAAsMAwsCfyAFKgIIu/0UIAUqAhi7/SIB/QwAAAAAAECPQAAAAAAAQI9A/fIBIhH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIQ4CfyAR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyECAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQIgAv0RIA79HAEgBf0cAiESIAwhBQNAIAMgBUECdCICaiABIAAgAmooAgBBBHRq/QAAACAS/bUBIhH9GwAgEf0bAWogEf0bAmoiAjYCACACIAogAiAKSBshCiACIA0gAiANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEBA0AgAyABQQJ0IgVqAn8gAiAAIAVqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAFBAWoiASALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIRcgBSoCGCEYIAUqAgghGUH4////ByEKQYiAgIB4IQ0gDCEFA0ACfyAXIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCAZIAIqAgCUIBggAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIUaigCAEECdCIVaigCACIORwRAIAX9CQI4IAggDkEGdGoiD/0JAgwgDyoCHP0gASAPKgIs/SACIA8qAjz9IAP95gEgBf0JAiggD/0JAgggDyoCGP0gASAPKgIo/SACIA8qAjj9IAP95gEgBf0JAgggD/0JAgAgDyoCEP0gASAPKgIg/SACIA8qAjD9IAP95gEgBf0JAhggD/0JAgQgDyoCFP0gASAPKgIk/SACIA8qAjT9IAP95gH95AH95AH95AEhESAOIQ8LIAMgFGoCfyAR/R8DIAEgFUECdCIOQQxyaioCAJQgEf0fAiABIA5BCHJqKgIAlCAR/R8AIAEgDmoqAgCUIBH9HwEgASAOQQRyaioCAJSSkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSACQQFqIgIgC0cNAAsMAQtBiICAgHghDUH4////ByEKCyALIAxLBEAgCUEBa7MgDbIgCrKTlSEXIAwhDQNAAn8gFyADIA1BAnRqIgEoAgAgCmuylCIYi0MAAABPXQRAIBioDAELQYCAgIB4CyEOIAEgDjYCACAEIA5BAnRqIgEgASgCAEEBajYCACANQQFqIg0gC0cNAAsLIAlBAk8EQCAEKAIAIQ1BASEKA0AgBCAKQQJ0aiIBIAEoAgAgDWoiDTYCACAKQQFqIgogCUcNAAsLIAxBAEoEQCAMIQoDQCAGIApBAWsiAUECdCICaiAAIAJqKAIANgIAIApBAUshAiABIQogAg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCwsEAEEACw==",jm="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACEgEDZW52Bm1lbW9yeQIDAICABAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=",GE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQrrDwICAAvlDwQBfAN7B30DfyALIAprIQwCQAJAIA4EQCANBEBB+P///wchCkGIgICAeCENIAsgDE0NAyAMIQUDQCADIAVBAnQiAWogAiAAIAFqKAIAQQJ0aigCACIBNgIAIAEgCiABIApIGyEKIAEgDSABIA1KGyENIAVBAWoiBSALRw0ACwwDCyAPBEAgCyAMTQ0CQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIcaigCACIdQQJ0aigCACIbRwRAAn8gBf0JAjggCCAbQQZ0aiIO/QkCDCAOKgIc/SABIA4qAiz9IAIgDioCPP0gA/3mASAF/QkCKCAO/QkCCCAOKgIY/SABIA4qAij9IAIgDioCOP0gA/3mASAF/QkCCCAO/QkCACAOKgIQ/SABIA4qAiD9IAIgDioCMP0gA/3mASAF/QkCGCAO/QkCBCAOKgIU/SABIA4qAiT9IAIgDioCNP0gA/3mAf3kAf3kAf3kASIR/V/9DAAAAAAAQI9AAAAAAABAj0AiEv3yASIT/SEBIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOAn8gE/0hACIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAv9ESAO/RwBAn8gESAR/Q0ICQoLDA0ODwABAgMAAQID/V8gEv3yASIR/SEAIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4C/0cAgJ/IBH9IQEiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgL/RwDIRIgGyEPCyADIBxqIAEgHUEEdGr9AAAAIBL9tQEiEf0bACAR/RsBaiAR/RsCaiAR/RsDaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAgi7/RQgBSoCGLv9IgH9DAAAAAAAQI9AAAAAAABAj0D98gEiEf0hASIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDgJ/IBH9IQAiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLAn8gBSoCKLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEFQfj///8HIQpBiICAgHghDSALIAxNDQL9ESAO/RwBIAX9HAIhEiAMIQUDQCADIAVBAnQiAmogASAAIAJqKAIAQQR0av0AAAAgEv21ASIR/RsAIBH9GwFqIBH9GwJqIgI2AgAgAiAKIAIgCkgbIQogAiANIAIgDUobIQ0gBUEBaiIFIAtHDQALDAILIA0EQEH4////ByEKQYiAgIB4IQ0gCyAMTQ0CIAwhBQNAIAMgBUECdCIBagJ/IAIgACABaigCAEECdGoqAgC7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAsiDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgD0UEQCALIAxNDQEgBSoCKCEUIAUqAhghFSAFKgIIIRZB+P///wchCkGIgICAeCENIAwhBQNAAn8gFCABIAAgBUECdCIHaigCAEEEdGoiAioCCJQgFiACKgIAlCAVIAIqAgSUkpK7RAAAAAAAALBAoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshDiADIAdqIA42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gBUEBaiIFIAtHDQALDAILIAsgDE0NAEF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiG2ooAgBBAnQiHGooAgAiDkcEQCAFKgI4IhQgCCAOQQZ0aiIPKgI8lCAFKgIoIhUgDyoCOJQgBSoCCCIWIA8qAjCUIAUqAhgiFyAPKgI0lJKSkiEYIBQgDyoCLJQgFSAPKgIolCAWIA8qAiCUIBcgDyoCJJSSkpIhGSAUIA8qAhyUIBUgDyoCGJQgFiAPKgIQlCAXIA8qAhSUkpKSIRogFCAPKgIMlCAVIA8qAgiUIBYgDyoCAJQgFyAPKgIElJKSkiEUIA4hDwsgAyAbagJ/IBggASAcQQJ0aiIOKgIMlCAZIA4qAgiUIBQgDioCAJQgGiAOKgIElJKSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAJBAWoiAiALRw0ACwwBC0GIgICAeCENQfj///8HIQoLIAsgDEsEQCAJQQFrsyANsiAKspOVIRQgDCENA0ACfyAUIAMgDUECdGoiASgCACAKa7KUIhWLQwAAAE9dBEAgFagMAQtBgICAgHgLIQ4gASAONgIAIAQgDkECdGoiASABKAIAQQFqNgIAIA1BAWoiDSALRw0ACwsgCUECTwRAIAQoAgAhDUEBIQoDQCAEIApBAnRqIgEgASgCACANaiINNgIAIApBAWoiCiAJRw0ACwsgDEEASgRAIAwhCgNAIAYgCkEBayIBQQJ0IgJqIAAgAmooAgA2AgAgCkEBSyABIQoNAAsLIAsgDEoEQCALIQoDQCAGIAsgBCADIApBAWsiCkECdCIBaigCAEECdGoiAigCACIFa0ECdGogACABaigCADYCACACIAVBAWs2AgAgCiAMSg0ACwsL",WE="AGFzbQEAAAAADwhkeWxpbmsuMAEEAAAAAAEXAmAAAGAQf39/f39/f39/f39/f39/fwACDwEDZW52Bm1lbW9yeQIAAAMDAgABBz4DEV9fd2FzbV9jYWxsX2N0b3JzAAAYX193YXNtX2FwcGx5X2RhdGFfcmVsb2NzAAALc29ydEluZGV4ZXMAAQqiDwICAAucDwMBfAd9Bn8gCyAKayEMAkACQCAOBEAgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQMgDCEFA0AgAyAFQQJ0IgFqIAIgACABaigCAEECdGooAgAiATYCACABIAogASAKSBshCiABIA0gASANShshDSAFQQFqIgUgC0cNAAsMAwsgDwRAIAsgDE0NAkF/IQ9B+P///wchCkGIgICAeCENIAwhAgNAIA8gByAAIAJBAnQiGmooAgBBAnQiG2ooAgAiDkcEQAJ/IAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRgCfyARIA8qAiyUIBIgDyoCKJQgEyAPKgIglCAUIA8qAiSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRkCfyARIA8qAhyUIBIgDyoCGJQgEyAPKgIQlCAUIA8qAhSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIRwCfyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSu0QAAAAAAECPQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIR0gDiEPCyADIBpqIAEgG0ECdGoiDigCBCAcbCAOKAIAIB1saiAOKAIIIBlsaiAOKAIMIBhsaiIONgIAIA4gCiAKIA5KGyEKIA4gDSANIA5IGyENIAJBAWoiAiALRw0ACwwDCwJ/IAUqAii7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshAgJ/IAUqAhi7RAAAAAAAQI9AoiIQmUQAAAAAAADgQWMEQCAQqgwBC0GAgICAeAshByALIAxNAn8gBSoCCLtEAAAAAABAj0CiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEPQfj///8HIQpBiICAgHghDQ0CIAwhBQNAIAMgBUECdCIIaiABIAAgCGooAgBBBHRqIggoAgQgB2wgCCgCACAPbGogCCgCCCACbGoiCDYCACAIIAogCCAKSBshCiAIIA0gCCANShshDSAFQQFqIgUgC0cNAAsMAgsgDQRAQfj///8HIQpBiICAgHghDSALIAxNDQIgDCEFA0AgAyAFQQJ0IgFqAn8gAiAAIAFqKAIAQQJ0aioCALtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyIONgIAIAogDiAKIA5IGyEKIA0gDiANIA5KGyENIAVBAWoiBSALRw0ACwwCCyAPRQRAIAsgDE0NASAFKgIoIREgBSoCGCESIAUqAgghE0H4////ByEKQYiAgIB4IQ0gDCEFA0ACfyARIAEgACAFQQJ0IgdqKAIAQQR0aiICKgIIlCATIAIqAgCUIBIgAioCBJSSkrtEAAAAAAAAsECiIhCZRAAAAAAAAOBBYwRAIBCqDAELQYCAgIB4CyEOIAMgB2ogDjYCACAKIA4gCiAOSBshCiANIA4gDSAOShshDSAFQQFqIgUgC0cNAAsMAgsgCyAMTQ0AQX8hD0H4////ByEKQYiAgIB4IQ0gDCECA0AgDyAHIAAgAkECdCIYaigCAEECdCIZaigCACIORwRAIAUqAjgiESAIIA5BBnRqIg8qAjyUIAUqAigiEiAPKgI4lCAFKgIIIhMgDyoCMJQgBSoCGCIUIA8qAjSUkpKSIRUgESAPKgIslCASIA8qAiiUIBMgDyoCIJQgFCAPKgIklJKSkiEWIBEgDyoCHJQgEiAPKgIYlCATIA8qAhCUIBQgDyoCFJSSkpIhFyARIA8qAgyUIBIgDyoCCJQgEyAPKgIAlCAUIA8qAgSUkpKSIREgDiEPCyADIBhqAn8gFSABIBlBAnRqIg4qAgyUIBYgDioCCJQgESAOKgIAlCAXIA4qAgSUkpKSu0QAAAAAAACwQKIiEJlEAAAAAAAA4EFjBEAgEKoMAQtBgICAgHgLIg42AgAgCiAOIAogDkgbIQogDSAOIA0gDkobIQ0gAkEBaiICIAtHDQALDAELQYiAgIB4IQ1B+P///wchCgsgCyAMSwRAIAlBAWuzIA2yIAqyk5UhESAMIQ0DQAJ/IBEgAyANQQJ0aiIBKAIAIAprspQiEotDAAAAT10EQCASqAwBC0GAgICAeAshDiABIA42AgAgBCAOQQJ0aiIBIAEoAgBBAWo2AgAgDUEBaiINIAtHDQALCyAJQQJPBEAgBCgCACENQQEhCgNAIAQgCkECdGoiASABKAIAIA1qIg02AgAgCkEBaiIKIAlHDQALCyAMQQBKBEAgDCEKA0AgBiAKQQFrIgFBAnQiAmogACACaigCADYCACAKQQFLIAEhCg0ACwsgCyAMSgRAIAshCgNAIAYgCyAEIAMgCkEBayIKQQJ0IgFqKAIAQQJ0aiICKAIAIgVrQQJ0aiAAIAFqKAIANgIAIAIgBUEBazYCACAKIAxKDQALCws=";function XE(i){let e,t,n,s,r,o,a,l,c,u,f,d,h,x,p,g,m,_,v,A;function S(y,M,E,b,C,P,B){const N=performance.now();if(!n&&(new Uint32Array(t,a,C.byteLength/A.BytesPerInt).set(C),new Float32Array(t,u,B.byteLength/A.BytesPerFloat).set(B),b)){let G;s?G=new Int32Array(t,f,P.byteLength/A.BytesPerInt):G=new Float32Array(t,f,P.byteLength/A.BytesPerFloat),G.set(P)}g||(g=new Uint32Array(_)),new Float32Array(t,p,16).set(E),new Uint32Array(t,h,_).set(g),e.exports.sortIndexes(a,x,f,d,h,p,l,c,u,_,y,M,o,b,s,r);const k={sortDone:!0,splatSortCount:y,splatRenderCount:M,sortTime:0};if(!n){const F=new Uint32Array(t,l,M);(!m||m.length<M)&&(m=new Uint32Array(M)),m.set(F),k.sortedIndexes=m}const V=performance.now();k.sortTime=V-N,i.postMessage(k)}i.onmessage=y=>{if(y.data.centers)centers=y.data.centers,sceneIndexes=y.data.sceneIndexes,s?new Int32Array(t,x+y.data.range.from*A.BytesPerInt*4,y.data.range.count*4).set(new Int32Array(centers)):new Float32Array(t,x+y.data.range.from*A.BytesPerFloat*4,y.data.range.count*4).set(new Float32Array(centers)),r&&new Uint32Array(t,c+y.data.range.from*4,y.data.range.count).set(new Uint32Array(sceneIndexes)),v=y.data.range.from+y.data.range.count;else if(y.data.sort){const M=Math.min(y.data.sort.splatRenderCount||0,v),E=Math.min(y.data.sort.splatSortCount||0,v),b=y.data.sort.usePrecomputedDistances;let C,P,B;n||(C=y.data.sort.indexesToSort,B=y.data.sort.transforms,b&&(P=y.data.sort.precomputedDistances)),S(E,M,y.data.sort.modelViewProj,b,C,P,B)}else if(y.data.init){A=y.data.init.Constants,o=y.data.init.splatCount,n=y.data.init.useSharedMemory,s=y.data.init.integerBasedSort,r=y.data.init.dynamicMode,_=y.data.init.distanceMapRange,v=0;const M=s?A.BytesPerInt*4:A.BytesPerFloat*4,E=new Uint8Array(y.data.init.sorterWasmBytes),b=16*A.BytesPerFloat,C=o*A.BytesPerInt,P=o*M,B=b,N=s?o*A.BytesPerInt:o*A.BytesPerFloat,k=o*A.BytesPerInt,V=o*A.BytesPerInt,F=s?_*A.BytesPerInt*2:_*A.BytesPerFloat*2,q=r?o*A.BytesPerInt:0,G=r?A.MaxScenes*b:0,j=A.MemoryPageSize*32,ce=C+P+B+N+k+F+V+q+G+j,ye=Math.floor(ce/A.MemoryPageSize)+1,Ie={module:{},env:{memory:new WebAssembly.Memory({initial:ye,maximum:ye,shared:!0})}};WebAssembly.compile(E).then(Ne=>WebAssembly.instantiate(Ne,Ie)).then(Ne=>{e=Ne,a=0,x=a+C,p=x+P,f=p+B,d=f+N,h=d+k,l=h+F,c=l+V,u=c+q,t=Ie.env.memory.buffer,n?i.postMessage({sortSetupPhase1Complete:!0,indexesToSortBuffer:t,indexesToSortOffset:a,sortedIndexesBuffer:t,sortedIndexesOffset:l,precomputedDistancesBuffer:t,precomputedDistancesOffset:f,transformsBuffer:t,transformsOffset:u}):i.postMessage({sortSetupPhase1Complete:!0})})}}}function qE(i,e,t,n,s,r=Ot.DefaultSplatSortDistanceMapPrecision){const o=new Worker(URL.createObjectURL(new Blob(["(",XE.toString(),")(self)"],{type:"application/javascript"})));let a=VE;const l=ah()?ix():null;!t&&!e?(a=jm,l&&l.major<=16&&l.minor<4&&(a=WE)):t?e||l&&l.major<=16&&l.minor<4&&(a=GE):a=jm;const c=atob(a),u=new Uint8Array(c.length);for(let f=0;f<c.length;f++)u[f]=c.charCodeAt(f);return o.postMessage({init:{sorterWasmBytes:u.buffer,splatCount:i,useSharedMemory:e,integerBasedSort:n,dynamicMode:s,distanceMapRange:1<<r,Constants:{BytesPerFloat:Ot.BytesPerFloat,BytesPerInt:Ot.BytesPerInt,MemoryPageSize:Ot.MemoryPageSize,MaxScenes:Ot.MaxScenes}}}),o}const Dr={None:0,VR:1,AR:2};class Qo{static createButton(e,t={}){const n=document.createElement("button");function s(){let c=null;async function u(h){h.addEventListener("end",f),await e.xr.setSession(h),n.textContent="EXIT VR",c=h}function f(){c.removeEventListener("end",f),n.textContent="ENTER VR",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="ENTER VR";const d={...t,optionalFeatures:["local-floor","bounded-floor","layers",...t.optionalFeatures||[]]};n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-vr",d).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",d).then(u).catch(h=>{console.warn(h)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-vr",d).then(u).catch(h=>{console.warn(h)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="VR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="VR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="VRButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-vr").then(function(c){c?s():o(),c&&Qo.xrSessionIsGranted&&n.click()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}static registerSessionGrantedListener(){if(typeof navigator<"u"&&"xr"in navigator){if(/WebXRViewer\//i.test(navigator.userAgent))return;navigator.xr.addEventListener("sessiongranted",()=>{Qo.xrSessionIsGranted=!0})}}}Qo.xrSessionIsGranted=!1;Qo.registerSessionGrantedListener();class YE{static createButton(e,t={}){const n=document.createElement("button");function s(){if(t.domOverlay===void 0){const d=document.createElement("div");d.style.display="none",document.body.appendChild(d);const h=document.createElementNS("http://www.w3.org/2000/svg","svg");h.setAttribute("width",38),h.setAttribute("height",38),h.style.position="absolute",h.style.right="20px",h.style.top="20px",h.addEventListener("click",function(){c.end()}),d.appendChild(h);const x=document.createElementNS("http://www.w3.org/2000/svg","path");x.setAttribute("d","M 12,12 L 28,28 M 28,12 12,28"),x.setAttribute("stroke","#fff"),x.setAttribute("stroke-width",2),h.appendChild(x),t.optionalFeatures===void 0&&(t.optionalFeatures=[]),t.optionalFeatures.push("dom-overlay"),t.domOverlay={root:d}}let c=null;async function u(d){d.addEventListener("end",f),e.xr.setReferenceSpaceType("local"),await e.xr.setSession(d),n.textContent="STOP AR",t.domOverlay.root.style.display="",c=d}function f(){c.removeEventListener("end",f),n.textContent="START AR",t.domOverlay.root.style.display="none",c=null}n.style.display="",n.style.cursor="pointer",n.style.left="calc(50% - 50px)",n.style.width="100px",n.textContent="START AR",n.onmouseenter=function(){n.style.opacity="1.0"},n.onmouseleave=function(){n.style.opacity="0.5"},n.onclick=function(){c===null?navigator.xr.requestSession("immersive-ar",t).then(u):(c.end(),navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(d=>{console.warn(d)}))},navigator.xr.offerSession!==void 0&&navigator.xr.offerSession("immersive-ar",t).then(u).catch(d=>{console.warn(d)})}function r(){n.style.display="",n.style.cursor="auto",n.style.left="calc(50% - 75px)",n.style.width="150px",n.onmouseenter=null,n.onmouseleave=null,n.onclick=null}function o(){r(),n.textContent="AR NOT SUPPORTED"}function a(c){r(),console.warn("Exception when trying to call xr.isSessionSupported",c),n.textContent="AR NOT ALLOWED"}function l(c){c.style.position="absolute",c.style.bottom="20px",c.style.padding="12px 6px",c.style.border="1px solid #fff",c.style.borderRadius="4px",c.style.background="rgba(0,0,0,0.1)",c.style.color="#fff",c.style.font="normal 13px sans-serif",c.style.textAlign="center",c.style.opacity="0.5",c.style.outline="none",c.style.zIndex="999"}if("xr"in navigator)return n.id="ARButton",n.style.display="none",l(n),navigator.xr.isSessionSupported("immersive-ar").then(function(c){c?s():o()}).catch(a),n;{const c=document.createElement("a");return window.isSecureContext===!1?(c.href=document.location.href.replace(/^http:/,"https:"),c.innerHTML="WEBXR NEEDS HTTPS"):(c.href="https://immersiveweb.dev/",c.innerHTML="WEBXR NOT AVAILABLE"),c.style.left="calc(50% - 90px)",c.style.width="180px",c.style.textDecoration="none",l(c),c}}}const nf={Always:0,Never:2},QE=50,KE=.75,jE=15e5,$E=10,ZE=2.5,JE=60;class bo{constructor(e={}){if(e.cameraUp||(e.cameraUp=[0,1,0]),this.cameraUp=new U().fromArray(e.cameraUp),e.initialCameraPosition||(e.initialCameraPosition=[0,10,15]),this.initialCameraPosition=new U().fromArray(e.initialCameraPosition),e.initialCameraLookAt||(e.initialCameraLookAt=[0,0,0]),this.initialCameraLookAt=new U().fromArray(e.initialCameraLookAt),this.dropInMode=e.dropInMode||!1,(e.selfDrivenMode===void 0||e.selfDrivenMode===null)&&(e.selfDrivenMode=!0),this.selfDrivenMode=e.selfDrivenMode&&!this.dropInMode,this.selfDrivenUpdateFunc=this.selfDrivenUpdate.bind(this),e.useBuiltInControls===void 0&&(e.useBuiltInControls=!0),this.useBuiltInControls=e.useBuiltInControls,this.rootElement=e.rootElement,this.ignoreDevicePixelRatio=e.ignoreDevicePixelRatio||!1,this.devicePixelRatio=this.ignoreDevicePixelRatio?1:window.devicePixelRatio||1,this.halfPrecisionCovariancesOnGPU=e.halfPrecisionCovariancesOnGPU||!1,this.threeScene=e.threeScene,this.renderer=e.renderer,this.camera=e.camera,this.gpuAcceleratedSort=e.gpuAcceleratedSort||!1,(e.integerBasedSort===void 0||e.integerBasedSort===null)&&(e.integerBasedSort=!0),this.integerBasedSort=e.integerBasedSort,(e.sharedMemoryForWorkers===void 0||e.sharedMemoryForWorkers===null)&&(e.sharedMemoryForWorkers=!0),this.sharedMemoryForWorkers=e.sharedMemoryForWorkers,this.dynamicScene=!!e.dynamicScene,this.antialiased=e.antialiased||!1,this.kernel2DSize=e.kernel2DSize===void 0?.3:e.kernel2DSize,this.webXRMode=e.webXRMode||Dr.None,this.webXRMode!==Dr.None&&(this.gpuAcceleratedSort=!1),this.webXRActive=!1,this.webXRSessionInit=e.webXRSessionInit||{},this.renderMode=e.renderMode||nf.Always,this.sceneRevealMode=e.sceneRevealMode||Ua.Default,this.focalAdjustment=e.focalAdjustment||1,this.maxScreenSpaceSplatSize=e.maxScreenSpaceSplatSize||1024,this.logLevel=e.logLevel||Bo.None,this.sphericalHarmonicsDegree=e.sphericalHarmonicsDegree||0,this.enableOptionalEffects=e.enableOptionalEffects||!1,(e.enableSIMDInSort===void 0||e.enableSIMDInSort===null)&&(e.enableSIMDInSort=!0),this.enableSIMDInSort=e.enableSIMDInSort,(e.inMemoryCompressionLevel===void 0||e.inMemoryCompressionLevel===null)&&(e.inMemoryCompressionLevel=0),this.inMemoryCompressionLevel=e.inMemoryCompressionLevel,(e.optimizeSplatData===void 0||e.optimizeSplatData===null)&&(e.optimizeSplatData=!0),this.optimizeSplatData=e.optimizeSplatData,(e.freeIntermediateSplatData===void 0||e.freeIntermediateSplatData===null)&&(e.freeIntermediateSplatData=!1),this.freeIntermediateSplatData=e.freeIntermediateSplatData,ah()){const n=ix();n.major<17&&(this.enableSIMDInSort=!1),n.major<16&&(this.sharedMemoryForWorkers=!1)}(e.splatRenderMode===void 0||e.splatRenderMode===null)&&(e.splatRenderMode=Cs.ThreeD),this.splatRenderMode=e.splatRenderMode,this.sceneFadeInRateMultiplier=e.sceneFadeInRateMultiplier||1,this.splatSortDistanceMapPrecision=e.splatSortDistanceMapPrecision||Ot.DefaultSplatSortDistanceMapPrecision;const t=this.integerBasedSort?20:24;this.splatSortDistanceMapPrecision=jt(this.splatSortDistanceMapPrecision,10,t),this.onSplatMeshChangedCallback=null,this.createSplatMesh(),this.controls=null,this.perspectiveControls=null,this.orthographicControls=null,this.orthographicCamera=null,this.perspectiveCamera=null,this.showMeshCursor=!1,this.showControlPlane=!1,this.showInfo=!1,this.sceneHelper=null,this.sortWorker=null,this.sortRunning=!1,this.splatRenderCount=0,this.splatSortCount=0,this.lastSplatSortCount=0,this.sortWorkerIndexesToSort=null,this.sortWorkerSortedIndexes=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.preSortMessages=[],this.runAfterNextSort=[],this.selfDrivenModeRunning=!1,this.splatRenderReady=!1,this.raycaster=new CE,this.infoPanel=null,this.startInOrthographicMode=!1,this.currentFPS=0,this.lastSortTime=0,this.consecutiveRenderFrames=0,this.previousCameraTarget=new U,this.nextCameraTarget=new U,this.mousePosition=new Ke,this.mouseDownPosition=new Ke,this.mouseDownTime=null,this.resizeObserver=null,this.mouseMoveListener=null,this.mouseDownListener=null,this.mouseUpListener=null,this.keyDownListener=null,this.sortPromise=null,this.sortPromiseResolver=null,this.splatSceneDownloadPromises={},this.splatSceneDownloadAndBuildPromise=null,this.splatSceneRemovalPromise=null,this.loadingSpinner=new Ah(null,this.rootElement||document.body),this.loadingSpinner.hide(),this.loadingProgressBar=new vE(this.rootElement||document.body),this.loadingProgressBar.hide(),this.infoPanel=new AE(this.rootElement||document.body),this.infoPanel.hide(),this.usingExternalCamera=!!(this.dropInMode||this.camera),this.usingExternalRenderer=!!(this.dropInMode||this.renderer),this.initialized=!1,this.disposing=!1,this.disposed=!1,this.disposePromise=null,this.dropInMode||this.init()}createSplatMesh(){this.splatMesh=new Tn(this.splatRenderMode,this.dynamicScene,this.enableOptionalEffects,this.halfPrecisionCovariancesOnGPU,this.devicePixelRatio,this.gpuAcceleratedSort,this.integerBasedSort,this.antialiased,this.maxScreenSpaceSplatSize,this.logLevel,this.sphericalHarmonicsDegree,this.sceneFadeInRateMultiplier,this.kernel2DSize),this.splatMesh.frustumCulled=!1,this.onSplatMeshChangedCallback&&this.onSplatMeshChangedCallback()}init(){this.initialized||(this.rootElement||(this.usingExternalRenderer?this.rootElement=this.renderer.domElement||document.body:(this.rootElement=document.createElement("div"),this.rootElement.style.width="100%",this.rootElement.style.height="100%",this.rootElement.style.position="absolute",document.body.appendChild(this.rootElement))),this.setupCamera(),this.setupRenderer(),this.setupWebXR(this.webXRSessionInit),this.setupControls(),this.setupEventHandlers(),this.threeScene=this.threeScene||new Ay,this.sceneHelper=new La(this.threeScene),this.sceneHelper.setupMeshCursor(),this.sceneHelper.setupFocusMarker(),this.sceneHelper.setupControlPlane(),this.loadingProgressBar.setContainer(this.rootElement),this.loadingSpinner.setContainer(this.rootElement),this.infoPanel.setContainer(this.rootElement),this.initialized=!0)}setupCamera(){if(!this.usingExternalCamera){const e=new Ke;this.getRenderDimensions(e),this.perspectiveCamera=new Ti(QE,e.x/e.y,.1,1e3),this.orthographicCamera=new rh(e.x/-2,e.x/2,e.y/2,e.y/-2,.1,1e3),this.camera=this.startInOrthographicMode?this.orthographicCamera:this.perspectiveCamera,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt)}}setupRenderer(){if(!this.usingExternalRenderer){const e=new Ke;this.getRenderDimensions(e),this.renderer=new x1({antialias:!1,precision:"highp"}),this.renderer.setPixelRatio(this.devicePixelRatio),this.renderer.autoClear=!0,this.renderer.setClearColor(new vt(0),0),this.renderer.setSize(e.x,e.y),this.resizeObserver=new ResizeObserver(()=>{this.getRenderDimensions(e),this.renderer.setSize(e.x,e.y),this.forceRenderNextFrame()}),this.resizeObserver.observe(this.rootElement),this.rootElement.appendChild(this.renderer.domElement)}}setupWebXR(e){this.webXRMode&&(this.webXRMode===Dr.VR?this.rootElement.appendChild(Qo.createButton(this.renderer,e)):this.webXRMode===Dr.AR&&this.rootElement.appendChild(YE.createButton(this.renderer,e)),this.renderer.xr.addEventListener("sessionstart",t=>{this.webXRActive=!0}),this.renderer.xr.addEventListener("sessionend",t=>{this.webXRActive=!1}),this.renderer.xr.enabled=!0,this.camera.position.copy(this.initialCameraPosition),this.camera.up.copy(this.cameraUp).normalize(),this.camera.lookAt(this.initialCameraLookAt))}setupControls(){if(this.useBuiltInControls&&this.webXRMode===Dr.None){this.usingExternalCamera?this.camera.isOrthographicCamera?this.orthographicControls=new jl(this.camera,this.renderer.domElement):this.perspectiveControls=new jl(this.camera,this.renderer.domElement):(this.perspectiveControls=new jl(this.perspectiveCamera,this.renderer.domElement),this.orthographicControls=new jl(this.orthographicCamera,this.renderer.domElement));for(let e of[this.orthographicControls,this.perspectiveControls])e&&(e.listenToKeyEvents(window),e.rotateSpeed=.5,e.maxPolarAngle=Math.PI*.75,e.minPolarAngle=.1,e.enableDamping=!0,e.dampingFactor=.05,e.target.copy(this.initialCameraLookAt),e.update());this.controls=this.camera.isOrthographicCamera?this.orthographicControls:this.perspectiveControls,this.controls.update()}}setupEventHandlers(){this.useBuiltInControls&&this.webXRMode===Dr.None&&(this.mouseMoveListener=this.onMouseMove.bind(this),this.renderer.domElement.addEventListener("pointermove",this.mouseMoveListener,!1),this.mouseDownListener=this.onMouseDown.bind(this),this.renderer.domElement.addEventListener("pointerdown",this.mouseDownListener,!1),this.mouseUpListener=this.onMouseUp.bind(this),this.renderer.domElement.addEventListener("pointerup",this.mouseUpListener,!1),this.keyDownListener=this.onKeyDown.bind(this),window.addEventListener("keydown",this.keyDownListener,!1))}removeEventHandlers(){this.useBuiltInControls&&(this.renderer.domElement.removeEventListener("pointermove",this.mouseMoveListener),this.mouseMoveListener=null,this.renderer.domElement.removeEventListener("pointerdown",this.mouseDownListener),this.mouseDownListener=null,this.renderer.domElement.removeEventListener("pointerup",this.mouseUpListener),this.mouseUpListener=null,window.removeEventListener("keydown",this.keyDownListener),this.keyDownListener=null)}setRenderMode(e){this.renderMode=e}setActiveSphericalHarmonicsDegrees(e){this.splatMesh.material.uniforms.sphericalHarmonicsDegree.value=e,this.splatMesh.material.uniformsNeedUpdate=!0}onSplatMeshChanged(e){this.onSplatMeshChangedCallback=e}onKeyDown=(function(){const e=new U,t=new rt,n=new rt;return function(s){switch(e.set(0,0,-1),e.transformDirection(this.camera.matrixWorld),t.makeRotationAxis(e,Math.PI/128),n.makeRotationAxis(e,-Math.PI/128),s.code){case"KeyG":this.focalAdjustment+=.02,this.forceRenderNextFrame();break;case"KeyF":this.focalAdjustment-=.02,this.forceRenderNextFrame();break;case"ArrowLeft":this.camera.up.transformDirection(t);break;case"ArrowRight":this.camera.up.transformDirection(n);break;case"KeyC":this.showMeshCursor=!this.showMeshCursor;break;case"KeyU":this.showControlPlane=!this.showControlPlane;break;case"KeyI":this.showInfo=!this.showInfo,this.showInfo?this.infoPanel.show():this.infoPanel.hide();break;case"KeyO":this.usingExternalCamera||this.setOrthographicMode(!this.camera.isOrthographicCamera);break;case"KeyP":this.usingExternalCamera||this.splatMesh.setPointCloudModeEnabled(!this.splatMesh.getPointCloudModeEnabled());break;case"Equal":this.usingExternalCamera||this.splatMesh.setSplatScale(this.splatMesh.getSplatScale()+.05);break;case"Minus":this.usingExternalCamera||this.splatMesh.setSplatScale(Math.max(this.splatMesh.getSplatScale()-.05,0));break}}})();onMouseMove(e){this.mousePosition.set(e.offsetX,e.offsetY)}onMouseDown(){this.mouseDownPosition.copy(this.mousePosition),this.mouseDownTime=xo()}onMouseUp=(function(){const e=new Ke;return function(t){e.copy(this.mousePosition).sub(this.mouseDownPosition),xo()-this.mouseDownTime<.5&&e.length()<2&&this.onMouseClick(t)}})();onMouseClick(e){this.mousePosition.set(e.offsetX,e.offsetY),this.checkForFocalPointChange()}checkForFocalPointChange=(function(){const e=new Ke,t=new U,n=[];return function(){if(!this.transitioningCameraTarget&&(this.getRenderDimensions(e),n.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,e),this.raycaster.intersectSplatMesh(this.splatMesh,n),n.length>0)){const r=n[0].origin;t.copy(r).sub(this.camera.position),t.length()>KE&&(this.previousCameraTarget.copy(this.controls.target),this.nextCameraTarget.copy(r),this.transitioningCameraTarget=!0,this.transitioningCameraTargetStartTime=xo())}}})();getRenderDimensions(e){this.rootElement?(e.x=this.rootElement.offsetWidth,e.y=this.rootElement.offsetHeight):this.renderer.getSize(e)}setOrthographicMode(e){if(e===this.camera.isOrthographicCamera)return;const t=this.camera,n=e?this.orthographicCamera:this.perspectiveCamera;if(n.position.copy(t.position),n.up.copy(t.up),n.rotation.copy(t.rotation),n.quaternion.copy(t.quaternion),n.matrix.copy(t.matrix),this.camera=n,this.controls){const s=a=>{a.saveState(),a.reset()},r=this.controls,o=e?this.orthographicControls:this.perspectiveControls;s(o),s(r),o.target.copy(r.target),e?bo.setCameraZoomFromPosition(n,t,r):bo.setCameraPositionFromZoom(n,t,o),this.controls=o,this.camera.lookAt(this.controls.target)}}static setCameraPositionFromZoom=(function(){const e=new U;return function(t,n,s){const r=1/(n.zoom*.001);e.copy(s.target).sub(t.position).normalize().multiplyScalar(r).negate(),t.position.copy(s.target).add(e)}})();static setCameraZoomFromPosition=(function(){const e=new U;return function(t,n,s){const r=e.copy(s.target).sub(n.position).length();t.zoom=1/(r*.001)}})();updateSplatMesh=(function(){const e=new Ke;return function(){if(!this.splatMesh)return;if(this.splatMesh.getSplatCount()>0){this.splatMesh.updateVisibleRegionFadeDistance(this.sceneRevealMode),this.splatMesh.updateTransforms(),this.getRenderDimensions(e);const n=this.camera.projectionMatrix.elements[0]*.5*this.devicePixelRatio*e.x,s=this.camera.projectionMatrix.elements[5]*.5*this.devicePixelRatio*e.y,r=this.camera.isOrthographicCamera?1/this.devicePixelRatio:1,o=this.focalAdjustment*r,a=1/o;this.adjustForWebXRStereo(e),this.splatMesh.updateUniforms(e,n*o,s*o,this.camera.isOrthographicCamera,this.camera.zoom||1,a)}}})();adjustForWebXRStereo(e){if(this.camera&&this.webXRActive){const n=this.renderer.xr.getCamera().projectionMatrix.elements[0],s=this.camera.projectionMatrix.elements[0];e.x*=s/n}}isLoadingOrUnloading(){return Object.keys(this.splatSceneDownloadPromises).length>0||this.splatSceneDownloadAndBuildPromise!==null||this.splatSceneRemovalPromise!==null}isDisposingOrDisposed(){return this.disposing||this.disposed}addSplatSceneDownloadPromise(e){this.splatSceneDownloadPromises[e.id]=e}removeSplatSceneDownloadPromise(e){delete this.splatSceneDownloadPromises[e.id]}setSplatSceneDownloadAndBuildPromise(e){this.splatSceneDownloadAndBuildPromise=e}clearSplatSceneDownloadAndBuildPromise(){this.splatSceneDownloadAndBuildPromise=null}addSplatScene(e,t={}){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");t.progressiveLoad&&this.splatMesh.scenes&&this.splatMesh.scenes.length>0&&(console.log('addSplatScene(): "progressiveLoad" option ignore because there are multiple splat scenes'),t.progressiveLoad=!1);const n=t.format!==void 0&&t.format!==null?t.format:Hm(e),s=bo.isProgressivelyLoadable(n)&&t.progressiveLoad,r=t.showLoadingUI!==void 0&&t.showLoadingUI!==null?t.showLoadingUI:!0;let o=null;r&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=()=>{this.loadingProgressBar.hide(),this.loadingSpinner.removeAllTasks()},l=(p,g,m)=>{if(r)if(m===an.Downloading)if(p==100)this.loadingSpinner.setMessageForTask(o,"Download complete!");else if(s)this.loadingSpinner.setMessageForTask(o,"Downloading splats...");else{const _=g?`: ${g}`:"...";this.loadingSpinner.setMessageForTask(o,`Downloading${_}`)}else m===an.Processing&&this.loadingSpinner.setMessageForTask(o,"Processing splats...")};let c=!1,u=0;const f=(p,g)=>{r&&((p&&s||g&&!s)&&(this.loadingSpinner.removeTask(o),!g&&!c&&this.loadingProgressBar.show()),s&&(g?(c=!0,this.loadingProgressBar.hide()):this.loadingProgressBar.setProgress(u)))},d=(p,g,m)=>{u=p,l(p,g,m),t.onProgress&&t.onProgress(p,g,m)},h=(p,g,m)=>{!s&&t.onProgress&&t.onProgress(0,"0%",an.Processing);const _={rotation:t.rotation||t.orientation,position:t.position,scale:t.scale,splatAlphaRemovalThreshold:t.splatAlphaRemovalThreshold};return this.addSplatBuffers([p],[_],m,g&&r,r,s,s).then(()=>{!s&&t.onProgress&&t.onProgress(100,"100%",an.Processing),f(g,m)})};return(s?this.downloadAndBuildSingleSplatSceneProgressiveLoad.bind(this):this.downloadAndBuildSingleSplatSceneStandardLoad.bind(this))(e,n,t.splatAlphaRemovalThreshold,h.bind(this),d,a.bind(this),t.headers)}downloadAndBuildSingleSplatSceneStandardLoad(e,t,n,s,r,o,a){const l=this.downloadSplatSceneToSplatBuffer(e,n,r,!1,void 0,t,a),c=Qu(l.abortHandler);return l.then(u=>(this.removeSplatSceneDownloadPromise(l),s(u,!0,!0).then(()=>{c.resolve(),this.clearSplatSceneDownloadAndBuildPromise()}))).catch(u=>{o&&o(),this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(l),c.reject(this.updateError(u,`Viewer::addSplatScene -> Could not load file ${e}`))}),this.addSplatSceneDownloadPromise(l),this.setSplatSceneDownloadAndBuildPromise(c.promise),c.promise}downloadAndBuildSingleSplatSceneProgressiveLoad(e,t,n,s,r,o,a){let l=0,c=!1;const u=[],f=()=>{if(u.length>0&&!c&&!this.isDisposingOrDisposed()){c=!0;const g=u.shift();s(g.splatBuffer,g.firstBuild,g.finalBuild).then(()=>{c=!1,g.firstBuild?x.resolve():g.finalBuild&&(p.resolve(),this.clearSplatSceneDownloadAndBuildPromise()),u.length>0&&pi(()=>f())})}},d=(g,m)=>{this.isDisposingOrDisposed()||(m||u.length===0||g.getSplatCount()>u[0].splatBuffer.getSplatCount())&&(u.push({splatBuffer:g,firstBuild:l===0,finalBuild:m}),l++,f())},h=this.downloadSplatSceneToSplatBuffer(e,n,r,!0,d,t,a),x=Qu(h.abortHandler),p=Qu();return this.addSplatSceneDownloadPromise(h),this.setSplatSceneDownloadAndBuildPromise(p.promise),h.then(()=>{this.removeSplatSceneDownloadPromise(h)}).catch(g=>{this.clearSplatSceneDownloadAndBuildPromise(),this.removeSplatSceneDownloadPromise(h);const m=this.updateError(g,"Viewer::addSplatScene -> Could not load one or more scenes");x.reject(m),o&&o(m)}),x.promise}addSplatScenes(e,t=!0,n=void 0){if(this.isLoadingOrUnloading())throw new Error("Cannot add splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot add splat scene after dispose() is called.");const s=e.length,r=[];let o;t&&(this.loadingSpinner.removeAllTasks(),o=this.loadingSpinner.addTask("Downloading..."));const a=(f,d,h,x)=>{r[f]=d;let p=0;for(let g=0;g<s;g++)p+=r[g]||0;p=p/s,h=`${p.toFixed(2)}%`,t&&x===an.Downloading&&this.loadingSpinner.setMessageForTask(o,p==100?"Download complete!":`Downloading: ${h}`),n&&n(p,h,x)},l=[],c=[];for(let f=0;f<e.length;f++){const d=e[f],h=d.format!==void 0&&d.format!==null?d.format:Hm(d.path),x=this.downloadSplatSceneToSplatBuffer(d.path,d.splatAlphaRemovalThreshold,a.bind(this,f),!1,void 0,h,d.headers);l.push(x),c.push(x.promise)}const u=new Zs((f,d)=>{Promise.all(c).then(h=>{t&&this.loadingSpinner.removeTask(o),n&&n(0,"0%",an.Processing),this.addSplatBuffers(h,e,!0,t,t,!1,!1).then(()=>{n&&n(100,"100%",an.Processing),this.clearSplatSceneDownloadAndBuildPromise(),f()})}).catch(h=>{t&&this.loadingSpinner.removeTask(o),this.clearSplatSceneDownloadAndBuildPromise(),d(this.updateError(h,"Viewer::addSplatScenes -> Could not load one or more splat scenes."))}).finally(()=>{this.removeSplatSceneDownloadPromise(u)})},f=>{for(let d of l)d.abort(f)});return this.addSplatSceneDownloadPromise(u),this.setSplatSceneDownloadAndBuildPromise(u),u}downloadSplatSceneToSplatBuffer(e,t=1,n=void 0,s=!1,r=void 0,o,a){try{if(o===ei.Splat||o===ei.KSplat||o===ei.Ply){const l=s?!1:this.optimizeSplatData;if(o===ei.Splat)return vh.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,a);if(o===ei.KSplat)return Fa.loadFromURL(e,n,s,r,a);if(o===ei.Ply)return xh.loadFromURL(e,n,s,r,t,this.inMemoryCompressionLevel,l,this.sphericalHarmonicsDegree,a)}else if(o===ei.Spz)return _h.loadFromURL(e,n,t,this.inMemoryCompressionLevel,this.optimizeSplatData,this.sphericalHarmonicsDegree,a)}catch(l){throw this.updateError(l,null)}throw new Error(`Viewer::downloadSplatSceneToSplatBuffer -> File format not supported: ${e}`)}static isProgressivelyLoadable(e){return e===ei.Splat||e===ei.KSplat||e===ei.Ply}addSplatBuffers=(function(){return function(e,t=[],n=!0,s=!0,r=!0,o=!1,a=!1,l=!0){if(this.isDisposingOrDisposed())return Promise.resolve();let c=null;const u=()=>{c!==null&&(this.loadingSpinner.removeTask(c),c=null)};return this.splatRenderReady=!1,new Promise(f=>{s&&(c=this.loadingSpinner.addTask("Processing splats...")),pi(()=>{if(this.isDisposingOrDisposed())f();else{const d=this.addSplatBuffersToMesh(e,t,n,r,o,l),h=this.splatMesh.getMaxSplatCount();this.sortWorker&&this.sortWorker.maxSplatCount!==h&&this.disposeSortWorker(),this.gpuAcceleratedSort||this.preSortMessages.push({centers:d.centers.buffer,sceneIndexes:d.sceneIndexes.buffer,range:{from:d.from,to:d.to,count:d.count}}),(!this.sortWorker&&h>0?this.setupSortWorker(this.splatMesh):Promise.resolve()).then(()=>{this.isDisposingOrDisposed()||this.runSplatSort(!0,!0).then(p=>{!this.sortWorker||!p?(this.splatRenderReady=!0,u(),f()):(a?this.splatRenderReady=!0:this.runAfterNextSort.push(()=>{this.splatRenderReady=!0}),this.runAfterNextSort.push(()=>{u(),f()}))})})}},!0)})}})();addSplatBuffersToMesh=(function(){let e;return function(t,n,s=!0,r=!1,o=!1,a=!0){if(this.isDisposingOrDisposed())return;let l=[],c=[];o||(l=this.splatMesh.scenes.map(h=>h.splatBuffer)||[],c=this.splatMesh.sceneOptions?this.splatMesh.sceneOptions.map(h=>h):[]),l.push(...t),c.push(...n),this.renderer&&this.splatMesh.setRenderer(this.renderer);const u=h=>{if(this.isDisposingOrDisposed())return;const x=this.splatMesh.getSplatCount();r&&x>=jE&&!h&&!e&&(this.loadingSpinner.setMinimized(!0,!0),e=this.loadingSpinner.addTask("Optimizing data structures..."))},f=h=>{this.isDisposingOrDisposed()||h&&e&&(this.loadingSpinner.removeTask(e),e=null)},d=this.splatMesh.build(l,c,!0,s,u,f,a);return s&&this.freeIntermediateSplatData&&this.splatMesh.freeIntermediateSplatData(),d}})();setupSortWorker(e){if(!this.isDisposingOrDisposed())return new Promise(t=>{const n=this.integerBasedSort?Int32Array:Float32Array,s=e.getSplatCount(),r=e.getMaxSplatCount();this.sortWorker=qE(r,this.sharedMemoryForWorkers,this.enableSIMDInSort,this.integerBasedSort,this.splatMesh.dynamicMode,this.splatSortDistanceMapPrecision),this.sortWorker.onmessage=o=>{if(o.data.sortDone){if(this.sortRunning=!1,this.sharedMemoryForWorkers)this.splatMesh.updateRenderIndexes(this.sortWorkerSortedIndexes,o.data.splatRenderCount);else{const a=new Uint32Array(o.data.sortedIndexes.buffer,0,o.data.splatRenderCount);this.splatMesh.updateRenderIndexes(a,o.data.splatRenderCount)}this.lastSplatSortCount=this.splatSortCount,this.lastSortTime=o.data.sortTime,this.sortPromiseResolver(),this.sortPromiseResolver=null,this.forceRenderNextFrame(),this.runAfterNextSort.length>0&&(this.runAfterNextSort.forEach(a=>{a()}),this.runAfterNextSort.length=0)}else if(o.data.sortCanceled)this.sortRunning=!1;else if(o.data.sortSetupPhase1Complete){this.logLevel>=Bo.Info&&console.log("Sorting web worker WASM setup complete."),this.sharedMemoryForWorkers?(this.sortWorkerSortedIndexes=new Uint32Array(o.data.sortedIndexesBuffer,o.data.sortedIndexesOffset,r),this.sortWorkerIndexesToSort=new Uint32Array(o.data.indexesToSortBuffer,o.data.indexesToSortOffset,r),this.sortWorkerPrecomputedDistances=new n(o.data.precomputedDistancesBuffer,o.data.precomputedDistancesOffset,r),this.sortWorkerTransforms=new Float32Array(o.data.transformsBuffer,o.data.transformsOffset,Ot.MaxScenes*16)):(this.sortWorkerIndexesToSort=new Uint32Array(r),this.sortWorkerPrecomputedDistances=new n(r),this.sortWorkerTransforms=new Float32Array(Ot.MaxScenes*16));for(let a=0;a<s;a++)this.sortWorkerIndexesToSort[a]=a;if(this.sortWorker.maxSplatCount=r,this.logLevel>=Bo.Info){console.log("Sorting web worker ready.");const a=this.splatMesh.getSplatDataTextures(),l=a.covariances.size,c=a.centerColors.size;console.log("Covariances texture size: "+l.x+" x "+l.y),console.log("Centers/colors texture size: "+c.x+" x "+c.y)}t()}}})}updateError(e,t){return e instanceof nx?e:e instanceof bc?new Error("File type or server does not support progressive loading."):t?new Error(t):e}disposeSortWorker(){this.sortWorker&&this.sortWorker.terminate(),this.sortWorker=null,this.sortPromise=null,this.sortPromiseResolver&&(this.sortPromiseResolver(),this.sortPromiseResolver=null),this.preSortMessages=[],this.sortRunning=!1}removeSplatScene(e,t=!0){return this.removeSplatScenes([e],t)}removeSplatScenes(e,t=!0){if(this.isLoadingOrUnloading())throw new Error("Cannot remove splat scene while another load or unload is already in progress.");if(this.isDisposingOrDisposed())throw new Error("Cannot remove splat scene after dispose() is called.");let n;return this.splatSceneRemovalPromise=new Promise((s,r)=>{let o;t&&(this.loadingSpinner.removeAllTasks(),this.loadingSpinner.show(),o=this.loadingSpinner.addTask("Removing splat scene..."));const a=()=>{t&&(this.loadingSpinner.hide(),this.loadingSpinner.removeTask(o))},l=u=>{a(),this.splatSceneRemovalPromise=null,u?r(u):s()},c=()=>this.isDisposingOrDisposed()?(l(),!0):!1;n=this.sortPromise||Promise.resolve(),n.then(()=>{if(c())return;const u=[],f=[],d=[];for(let h=0;h<this.splatMesh.scenes.length;h++){let x=!1;for(let p of e)if(p===h){x=!0;break}if(!x){const p=this.splatMesh.scenes[h];u.push(p.splatBuffer),f.push(this.splatMesh.sceneOptions[h]),d.push({position:p.position.clone(),quaternion:p.quaternion.clone(),scale:p.scale.clone()})}}this.disposeSortWorker(),this.splatMesh.dispose(),this.sceneRevealMode=Ua.Instant,this.createSplatMesh(),this.addSplatBuffers(u,f,!0,!1,!0).then(()=>{c()||(a(),this.splatMesh.scenes.forEach((h,x)=>{h.position.copy(d[x].position),h.quaternion.copy(d[x].quaternion),h.scale.copy(d[x].scale)}),this.splatMesh.updateTransforms(),this.splatRenderReady=!1,this.runSplatSort(!0).then(()=>{if(c()){this.splatRenderReady=!0;return}n=this.sortPromise||Promise.resolve(),n.then(()=>{this.splatRenderReady=!0,l()})}))}).catch(h=>{l(h)})})}),this.splatSceneRemovalPromise}start(){if(this.selfDrivenMode)this.webXRMode?this.renderer.setAnimationLoop(this.selfDrivenUpdateFunc):this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc),this.selfDrivenModeRunning=!0;else throw new Error("Cannot start viewer unless it is in self driven mode.")}stop(){this.selfDrivenMode&&this.selfDrivenModeRunning&&(this.webXRMode?this.renderer.setAnimationLoop(null):cancelAnimationFrame(this.requestFrameId),this.selfDrivenModeRunning=!1)}async dispose(){if(this.isDisposingOrDisposed())return this.disposePromise;let e=[],t=[];for(let n in this.splatSceneDownloadPromises)if(this.splatSceneDownloadPromises.hasOwnProperty(n)){const s=this.splatSceneDownloadPromises[n];t.push(s),e.push(s.promise)}return this.sortPromise&&e.push(this.sortPromise),this.disposing=!0,this.disposePromise=Promise.all(e).finally(()=>{this.stop(),this.orthographicControls&&(this.orthographicControls.dispose(),this.orthographicControls=null),this.perspectiveControls&&(this.perspectiveControls.dispose(),this.perspectiveControls=null),this.controls=null,this.splatMesh&&(this.splatMesh.dispose(),this.splatMesh=null),this.sceneHelper&&(this.sceneHelper.dispose(),this.sceneHelper=null),this.resizeObserver&&(this.resizeObserver.unobserve(this.rootElement),this.resizeObserver=null),this.disposeSortWorker(),this.removeEventHandlers(),this.loadingSpinner.removeAllTasks(),this.loadingSpinner.setContainer(null),this.loadingProgressBar.hide(),this.loadingProgressBar.setContainer(null),this.infoPanel.setContainer(null),this.camera=null,this.threeScene=null,this.splatRenderReady=!1,this.initialized=!1,this.renderer&&(this.usingExternalRenderer||(this.rootElement.removeChild(this.renderer.domElement),this.renderer.dispose()),this.renderer=null),this.usingExternalRenderer||document.body.removeChild(this.rootElement),this.sortWorkerSortedIndexes=null,this.sortWorkerIndexesToSort=null,this.sortWorkerPrecomputedDistances=null,this.sortWorkerTransforms=null,this.disposed=!0,this.disposing=!1,this.disposePromise=null}),t.forEach(n=>{n.abort("Scene disposed")}),this.disposePromise}selfDrivenUpdate(){this.selfDrivenMode&&!this.webXRMode&&(this.requestFrameId=requestAnimationFrame(this.selfDrivenUpdateFunc)),this.update(),this.shouldRender()?(this.render(),this.consecutiveRenderFrames++):this.consecutiveRenderFrames=0,this.renderNextFrame=!1}forceRenderNextFrame(){this.renderNextFrame=!0}shouldRender=(function(){let e=0;const t=new U,n=new Gt,s=1e-4;return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return!1;let r=!1,o=!1;if(this.camera){const a=this.camera.position,l=this.camera.quaternion;o=Math.abs(a.x-t.x)>s||Math.abs(a.y-t.y)>s||Math.abs(a.z-t.z)>s||Math.abs(l.x-n.x)>s||Math.abs(l.y-n.y)>s||Math.abs(l.z-n.z)>s||Math.abs(l.w-n.w)>s}return r=this.renderMode!==nf.Never&&(e===0||this.splatMesh.visibleRegionChanging||o||this.renderMode===nf.Always||this.dynamicMode===!0||this.renderNextFrame),this.camera&&(t.copy(this.camera.position),n.copy(this.camera.quaternion)),e++,r}})();render=(function(){return function(){if(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())return;const e=n=>{for(let s of n.children)if(s.visible)return!0;return!1},t=this.renderer.autoClear;e(this.threeScene)&&(this.renderer.render(this.threeScene,this.camera),this.renderer.autoClear=!1),this.renderer.render(this.splatMesh,this.camera),this.renderer.autoClear=!1,this.sceneHelper.getFocusMarkerOpacity()>0&&this.renderer.render(this.sceneHelper.focusMarker,this.camera),this.showControlPlane&&this.renderer.render(this.sceneHelper.controlPlane,this.camera),this.renderer.autoClear=t}})();update(e,t){this.dropInMode&&this.updateForDropInMode(e,t),!(!this.initialized||!this.splatRenderReady||this.isDisposingOrDisposed())&&(this.controls&&(this.controls.update(),this.camera.isOrthographicCamera&&!this.usingExternalCamera&&bo.setCameraPositionFromZoom(this.camera,this.camera,this.controls)),this.runSplatSort(),this.updateForRendererSizeChanges(),this.updateSplatMesh(),this.updateMeshCursor(),this.updateFPS(),this.timingSensitiveUpdates(),this.updateInfoPanel(),this.updateControlPlane())}updateForDropInMode(e,t){this.renderer=e,this.splatMesh&&this.splatMesh.setRenderer(this.renderer),this.camera=t,this.controls&&(this.controls.object=t),this.init()}updateFPS=(function(){let e=xo(),t=0;return function(){if(this.consecutiveRenderFrames>JE){const n=xo();n-e>=1?(this.currentFPS=t,t=0,e=n):t++}else this.currentFPS=null}})();updateForRendererSizeChanges=(function(){const e=new Ke,t=new Ke;let n;return function(){this.usingExternalCamera||(this.renderer.getSize(t),(n===void 0||n!==this.camera.isOrthographicCamera||t.x!==e.x||t.y!==e.y)&&(this.camera.isOrthographicCamera?(this.camera.left=-t.x/2,this.camera.right=t.x/2,this.camera.top=t.y/2,this.camera.bottom=-t.y/2):this.camera.aspect=t.x/t.y,this.camera.updateProjectionMatrix(),e.copy(t),n=this.camera.isOrthographicCamera))}})();timingSensitiveUpdates=(function(){let e;return function(){const t=xo();e||(e=t);const n=t-e;this.updateCameraTransition(t),this.updateFocusMarker(n),e=t}})();updateCameraTransition=(function(){let e=new U,t=new U,n=new U;return function(s){if(this.transitioningCameraTarget){t.copy(this.previousCameraTarget).sub(this.camera.position).normalize(),n.copy(this.nextCameraTarget).sub(this.camera.position).normalize();const r=Math.acos(t.dot(n)),a=(r/(Math.PI/3)*.65+.3)/r*(s-this.transitioningCameraTargetStartTime);e.copy(this.previousCameraTarget).lerp(this.nextCameraTarget,a),this.camera.lookAt(e),this.controls.target.copy(e),a>=1&&(this.transitioningCameraTarget=!1)}}})();updateFocusMarker=(function(){const e=new Ke;let t=!1;return function(n){if(this.getRenderDimensions(e),this.transitioningCameraTarget){this.sceneHelper.setFocusMarkerVisibility(!0);const s=Math.max(this.sceneHelper.getFocusMarkerOpacity(),0);let r=Math.min(s+$E*n,1);this.sceneHelper.setFocusMarkerOpacity(r),this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e),t=!0,this.forceRenderNextFrame()}else{let s;if(t?s=1:s=Math.min(this.sceneHelper.getFocusMarkerOpacity(),1),s>0){this.sceneHelper.updateFocusMarker(this.nextCameraTarget,this.camera,e);let r=Math.max(s-ZE*n,0);this.sceneHelper.setFocusMarkerOpacity(r),r===0&&this.sceneHelper.setFocusMarkerVisibility(!1)}s>0&&this.forceRenderNextFrame(),t=!1}}})();updateMeshCursor=(function(){const e=[],t=new Ke;return function(){this.showMeshCursor?(this.forceRenderNextFrame(),this.getRenderDimensions(t),e.length=0,this.raycaster.setFromCameraAndScreenPosition(this.camera,this.mousePosition,t),this.raycaster.intersectSplatMesh(this.splatMesh,e),e.length>0?(this.sceneHelper.setMeshCursorVisibility(!0),this.sceneHelper.positionAndOrientMeshCursor(e[0].origin,this.camera)):this.sceneHelper.setMeshCursorVisibility(!1)):(this.sceneHelper.getMeschCursorVisibility()&&this.forceRenderNextFrame(),this.sceneHelper.setMeshCursorVisibility(!1))}})();updateInfoPanel=(function(){const e=new Ke;return function(){if(!this.showInfo)return;const t=this.splatMesh.getSplatCount();this.getRenderDimensions(e);const n=this.controls?this.controls.target:null,s=this.showMeshCursor?this.sceneHelper.meshCursor.position:null,r=t>0?this.splatRenderCount/t*100:0;this.infoPanel.update(e,this.camera.position,n,this.camera.up,this.camera.isOrthographicCamera,s,this.currentFPS||"N/A",t,this.splatRenderCount,r,this.lastSortTime,this.focalAdjustment,this.splatMesh.getSplatScale(),this.splatMesh.getPointCloudModeEnabled())}})();updateControlPlane(){this.showControlPlane?(this.sceneHelper.setControlPlaneVisibility(!0),this.sceneHelper.positionAndOrientControlPlane(this.controls.target,this.camera.up)):this.sceneHelper.setControlPlaneVisibility(!1)}runSplatSort=(function(){const e=new rt,t=[],n=new U(0,0,-1),s=new U(0,0,-1),r=new U,o=new U,a=[],l=[{angleThreshold:.55,sortFractions:[.125,.33333,.75]},{angleThreshold:.65,sortFractions:[.33333,.66667]},{angleThreshold:.8,sortFractions:[.5]}];return function(c=!1,u=!1){if(!this.initialized)return Promise.resolve(!1);if(this.sortRunning)return Promise.resolve(!0);if(this.splatMesh.getSplatCount()<=0)return this.splatRenderCount=0,Promise.resolve(!1);let f=0,d=0,h=!1,x=!1;if(s.set(0,0,-1).applyQuaternion(this.camera.quaternion),f=s.dot(n),d=o.copy(this.camera.position).sub(r).length(),!c&&!this.splatMesh.dynamicMode&&a.length===0&&(f<=.99&&(h=!0),d>=1&&(x=!0),!h&&!x))return Promise.resolve(!1);this.sortRunning=!0;let{splatRenderCount:p,shouldSortAll:g}=this.gatherSceneNodesForSort();g=g||u,this.splatRenderCount=p,e.copy(this.camera.matrixWorld).invert();const m=this.perspectiveCamera||this.camera;e.premultiply(m.projectionMatrix),this.splatMesh.dynamicMode||e.multiply(this.splatMesh.matrixWorld);let _=Promise.resolve(!0);return this.gpuAcceleratedSort&&(a.length<=1||a.length%2===0)&&(_=this.splatMesh.computeDistancesOnGPU(e,this.sortWorkerPrecomputedDistances)),_.then(()=>{if(a.length===0)if(this.splatMesh.dynamicMode||g)a.push(this.splatRenderCount);else{for(let S of l)if(f<S.angleThreshold){for(let y of S.sortFractions)a.push(Math.floor(this.splatRenderCount*y));break}a.push(this.splatRenderCount)}let v=Math.min(a.shift(),this.splatRenderCount);this.splatSortCount=v,t[0]=this.camera.position.x,t[1]=this.camera.position.y,t[2]=this.camera.position.z;const A={modelViewProj:e.elements,cameraPosition:t,splatRenderCount:this.splatRenderCount,splatSortCount:v,usePrecomputedDistances:this.gpuAcceleratedSort};return this.splatMesh.dynamicMode&&this.splatMesh.fillTransformsArray(this.sortWorkerTransforms),this.sharedMemoryForWorkers||(A.indexesToSort=this.sortWorkerIndexesToSort,A.transforms=this.sortWorkerTransforms,this.gpuAcceleratedSort&&(A.precomputedDistances=this.sortWorkerPrecomputedDistances)),this.sortPromise=new Promise(S=>{this.sortPromiseResolver=S}),this.preSortMessages.length>0&&(this.preSortMessages.forEach(S=>{this.sortWorker.postMessage(S)}),this.preSortMessages=[]),this.sortWorker.postMessage({sort:A}),a.length===0&&(r.copy(this.camera.position),n.copy(s)),!0}),_}})();gatherSceneNodesForSort=(function(){const e=[];let t=null;const n=new U,s=new U,r=new U,o=new rt,a=new rt,l=new rt,c=new U,u=new U(0,0,-1),f=new U,d=h=>f.copy(h.max).sub(h.min).length();return function(h=!1){this.getRenderDimensions(c);const x=c.y/2/Math.tan(this.camera.fov/2*An.DEG2RAD),p=Math.atan(c.x/2/x),g=Math.atan(c.y/2/x),m=Math.cos(p),_=Math.cos(g),v=this.splatMesh.getSplatTree();if(v){a.copy(this.camera.matrixWorld).invert(),this.splatMesh.dynamicMode||a.multiply(this.splatMesh.matrixWorld);let A=0,S=0;for(let M=0;M<v.subTrees.length;M++){const E=v.subTrees[M];o.copy(a),this.splatMesh.dynamicMode&&(this.splatMesh.getSceneTransform(M,l),o.multiply(l));const b=E.nodesWithIndexes.length;for(let C=0;C<b;C++){const P=E.nodesWithIndexes[C];if(!P.data||!P.data.indexes||P.data.indexes.length===0)continue;r.copy(P.center).applyMatrix4(o);const B=r.length();r.normalize(),n.copy(r).setX(0).normalize(),s.copy(r).setY(0).normalize();const N=u.dot(s),k=u.dot(n),V=d(P),F=k<_-.6,q=N<m-.6;!h&&(q||F)&&B>V||(S+=P.data.indexes.length,e[A]=P,P.data.distanceToNode=B,A++)}}e.length=A,e.sort((M,E)=>M.data.distanceToNode<E.data.distanceToNode?-1:1);let y=S*Ot.BytesPerInt;for(let M=0;M<A;M++){const E=e[M],b=E.data.indexes.length,C=b*Ot.BytesPerInt;new Uint32Array(this.sortWorkerIndexesToSort.buffer,y-C,b).set(E.data.indexes),y-=C}return{splatRenderCount:S,shouldSortAll:!1}}else{const A=this.splatMesh.getSplatCount();if(!t||t.length!==A){t=new Uint32Array(A);for(let S=0;S<A;S++)t[S]=S}return this.sortWorkerIndexesToSort.set(t),{splatRenderCount:A,shouldSortAll:!0}}}})();getSplatMesh(){return this.splatMesh}getSplatScene(e){return this.splatMesh.getScene(e)}getSceneCount(){return this.splatMesh.getSceneCount()}isMobile(){return navigator.userAgent.includes("Mobi")}}function bs(i){if(i===void 0)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return i}function fx(i,e){i.prototype=Object.create(e.prototype),i.prototype.constructor=i,i.__proto__=e}var _i={autoSleep:120,force3D:"auto",nullTargetWarn:1,units:{lineHeight:""}},Ko={duration:.5,overwrite:!1,delay:0},bh,wn,$t,Di=1e8,Ht=1/Di,pd=Math.PI*2,ew=pd/4,tw=0,dx=Math.sqrt,nw=Math.cos,iw=Math.sin,Sn=function(e){return typeof e=="string"},rn=function(e){return typeof e=="function"},Us=function(e){return typeof e=="number"},Mh=function(e){return typeof e>"u"},cs=function(e){return typeof e=="object"},ni=function(e){return e!==!1},Ch=function(){return typeof window<"u"},Zl=function(e){return rn(e)||Sn(e)},hx=typeof ArrayBuffer=="function"&&ArrayBuffer.isView||function(){},Un=Array.isArray,sw=/random\([^)]+\)/g,rw=/,\s*/g,$m=/(?:-?\.?\d|\.)+/gi,px=/[-+=.]*\d+[.e\-+]*\d*[e\-+]*\d*/g,Mo=/[-+=.]*\d+[.e-]*\d*[a-z%]*/g,sf=/[-+=.]*\d+\.?\d*(?:e-|e\+)?\d*/gi,mx=/[+-]=-?[.\d]+/,ow=/[^,'"\[\]\s]+/gi,aw=/^[+\-=e\s\d]*\d+[.\d]*([a-z]*|%)\s*$/i,tn,Qi,md,Th,vi={},Tc={},gx,xx=function(e){return(Tc=jo(e,vi))&&oi},Eh=function(e,t){return console.warn("Invalid property",e,"set to",t,"Missing plugin? gsap.registerPlugin()")},el=function(e,t){return!t&&console.warn(e)},_x=function(e,t){return e&&(vi[e]=t)&&Tc&&(Tc[e]=t)||vi},tl=function(){return 0},lw={suppressEvents:!0,isStart:!0,kill:!1},cc={suppressEvents:!0,kill:!1},cw={suppressEvents:!0},wh={},rr=[],gd={},vx,fi={},rf={},Zm=30,uc=[],Rh="",Ih=function(e){var t=e[0],n,s;if(cs(t)||rn(t)||(e=[e]),!(n=(t._gsap||{}).harness)){for(s=uc.length;s--&&!uc[s].targetTest(t););n=uc[s]}for(s=e.length;s--;)e[s]&&(e[s]._gsap||(e[s]._gsap=new Gx(e[s],n)))||e.splice(s,1);return e},Gr=function(e){return e._gsap||Ih(Pi(e))[0]._gsap},Ax=function(e,t,n){return(n=e[t])&&rn(n)?e[t]():Mh(n)&&e.getAttribute&&e.getAttribute(t)||n},ii=function(e,t){return(e=e.split(",")).forEach(t)||e},on=function(e){return Math.round(e*1e5)/1e5||0},en=function(e){return Math.round(e*1e7)/1e7||0},Uo=function(e,t){var n=t.charAt(0),s=parseFloat(t.substr(2));return e=parseFloat(e),n==="+"?e+s:n==="-"?e-s:n==="*"?e*s:e/s},uw=function(e,t){for(var n=t.length,s=0;e.indexOf(t[s])<0&&++s<n;);return s<n},Ec=function(){var e=rr.length,t=rr.slice(0),n,s;for(gd={},rr.length=0,n=0;n<e;n++)s=t[n],s&&s._lazy&&(s.render(s._lazy[0],s._lazy[1],!0)._lazy=0)},Dh=function(e){return!!(e._initted||e._startAt||e.add)},Sx=function(e,t,n,s){rr.length&&!wn&&Ec(),e.render(t,n,!!(wn&&t<0&&Dh(e))),rr.length&&!wn&&Ec()},yx=function(e){var t=parseFloat(e);return(t||t===0)&&(e+"").match(ow).length<2?t:Sn(e)?e.trim():e},bx=function(e){return e},Ai=function(e,t){for(var n in t)n in e||(e[n]=t[n]);return e},fw=function(e){return function(t,n){for(var s in n)s in t||s==="duration"&&e||s==="ease"||(t[s]=n[s])}},jo=function(e,t){for(var n in t)e[n]=t[n];return e},Jm=function i(e,t){for(var n in t)n!=="__proto__"&&n!=="constructor"&&n!=="prototype"&&(e[n]=cs(t[n])?i(e[n]||(e[n]={}),t[n]):t[n]);return e},wc=function(e,t){var n={},s;for(s in e)s in t||(n[s]=e[s]);return n},Oa=function(e){var t=e.parent||tn,n=e.keyframes?fw(Un(e.keyframes)):Ai;if(ni(e.inherit))for(;t;)n(e,t.vars.defaults),t=t.parent||t._dp;return e},dw=function(e,t){for(var n=e.length,s=n===t.length;s&&n--&&e[n]===t[n];);return n<0},Mx=function(e,t,n,s,r){var o=e[s],a;if(r)for(a=t[r];o&&o[r]>a;)o=o._prev;return o?(t._next=o._next,o._next=t):(t._next=e[n],e[n]=t),t._next?t._next._prev=t:e[s]=t,t._prev=o,t.parent=t._dp=e,t},Kc=function(e,t,n,s){n===void 0&&(n="_first"),s===void 0&&(s="_last");var r=t._prev,o=t._next;r?r._next=o:e[n]===t&&(e[n]=o),o?o._prev=r:e[s]===t&&(e[s]=r),t._next=t._prev=t.parent=null},ur=function(e,t){e.parent&&(!t||e.parent.autoRemoveChildren)&&e.parent.remove&&e.parent.remove(e),e._act=0},Wr=function(e,t){if(e&&(!t||t._end>e._dur||t._start<0))for(var n=e;n;)n._dirty=1,n=n.parent;return e},hw=function(e){for(var t=e.parent;t&&t.parent;)t._dirty=1,t.totalDuration(),t=t.parent;return e},xd=function(e,t,n,s){return e._startAt&&(wn?e._startAt.revert(cc):e.vars.immediateRender&&!e.vars.autoRevert||e._startAt.render(t,!0,s))},pw=function i(e){return!e||e._ts&&i(e.parent)},e0=function(e){return e._repeat?$o(e._tTime,e=e.duration()+e._rDelay)*e:0},$o=function(e,t){var n=Math.floor(e=en(e/t));return e&&n===e?n-1:n},Rc=function(e,t){return(e-t._start)*t._ts+(t._ts>=0?0:t._dirty?t.totalDuration():t._tDur)},jc=function(e){return e._end=en(e._start+(e._tDur/Math.abs(e._ts||e._rts||Ht)||0))},$c=function(e,t){var n=e._dp;return n&&n.smoothChildTiming&&e._ts&&(e._start=en(n._time-(e._ts>0?t/e._ts:((e._dirty?e.totalDuration():e._tDur)-t)/-e._ts)),jc(e),n._dirty||Wr(n,e)),e},Cx=function(e,t){var n;if((t._time||!t._dur&&t._initted||t._start<e._time&&(t._dur||!t.add))&&(n=Rc(e.rawTime(),t),(!t._dur||hl(0,t.totalDuration(),n)-t._tTime>Ht)&&t.render(n,!0)),Wr(e,t)._dp&&e._initted&&e._time>=e._dur&&e._ts){if(e._dur<e.duration())for(n=e;n._dp;)n.rawTime()>=0&&n.totalTime(n._tTime),n=n._dp;e._zTime=-Ht}},Zi=function(e,t,n,s){return t.parent&&ur(t),t._start=en((Us(n)?n:n||e!==tn?Mi(e,n,t):e._time)+t._delay),t._end=en(t._start+(t.totalDuration()/Math.abs(t.timeScale())||0)),Mx(e,t,"_first","_last",e._sort?"_start":0),_d(t)||(e._recent=t),s||Cx(e,t),e._ts<0&&$c(e,e._tTime),e},Tx=function(e,t){return(vi.ScrollTrigger||Eh("scrollTrigger",t))&&vi.ScrollTrigger.create(t,e)},Ex=function(e,t,n,s,r){if(Fh(e,t,r),!e._initted)return 1;if(!n&&e._pt&&!wn&&(e._dur&&e.vars.lazy!==!1||!e._dur&&e.vars.lazy)&&vx!==di.frame)return rr.push(e),e._lazy=[r,s],1},mw=function i(e){var t=e.parent;return t&&t._ts&&t._initted&&!t._lock&&(t.rawTime()<0||i(t))},_d=function(e){var t=e.data;return t==="isFromStart"||t==="isStart"},gw=function(e,t,n,s){var r=e.ratio,o=t<0||!t&&(!e._start&&mw(e)&&!(!e._initted&&_d(e))||(e._ts<0||e._dp._ts<0)&&!_d(e))?0:1,a=e._rDelay,l=0,c,u,f;if(a&&e._repeat&&(l=hl(0,e._tDur,t),u=$o(l,a),e._yoyo&&u&1&&(o=1-o),u!==$o(e._tTime,a)&&(r=1-o,e.vars.repeatRefresh&&e._initted&&e.invalidate())),o!==r||wn||s||e._zTime===Ht||!t&&e._zTime){if(!e._initted&&Ex(e,t,s,n,l))return;for(f=e._zTime,e._zTime=t||(n?Ht:0),n||(n=t&&!f),e.ratio=o,e._from&&(o=1-o),e._time=0,e._tTime=l,c=e._pt;c;)c.r(o,c.d),c=c._next;t<0&&xd(e,t,n,!0),e._onUpdate&&!n&&mi(e,"onUpdate"),l&&e._repeat&&!n&&e.parent&&mi(e,"onRepeat"),(t>=e._tDur||t<0)&&e.ratio===o&&(o&&ur(e,1),!n&&!wn&&(mi(e,o?"onComplete":"onReverseComplete",!0),e._prom&&e._prom()))}else e._zTime||(e._zTime=t)},xw=function(e,t,n){var s;if(n>t)for(s=e._first;s&&s._start<=n;){if(s.data==="isPause"&&s._start>t)return s;s=s._next}else for(s=e._last;s&&s._start>=n;){if(s.data==="isPause"&&s._start<t)return s;s=s._prev}},Zo=function(e,t,n,s){var r=e._repeat,o=en(t)||0,a=e._tTime/e._tDur;return a&&!s&&(e._time*=o/e._dur),e._dur=o,e._tDur=r?r<0?1e10:en(o*(r+1)+e._rDelay*r):o,a>0&&!s&&$c(e,e._tTime=e._tDur*a),e.parent&&jc(e),n||Wr(e.parent,e),e},t0=function(e){return e instanceof qn?Wr(e):Zo(e,e._dur)},_w={_start:0,endTime:tl,totalDuration:tl},Mi=function i(e,t,n){var s=e.labels,r=e._recent||_w,o=e.duration()>=Di?r.endTime(!1):e._dur,a,l,c;return Sn(t)&&(isNaN(t)||t in s)?(l=t.charAt(0),c=t.substr(-1)==="%",a=t.indexOf("="),l==="<"||l===">"?(a>=0&&(t=t.replace(/=/,"")),(l==="<"?r._start:r.endTime(r._repeat>=0))+(parseFloat(t.substr(1))||0)*(c?(a<0?r:n).totalDuration()/100:1)):a<0?(t in s||(s[t]=o),s[t]):(l=parseFloat(t.charAt(a-1)+t.substr(a+1)),c&&n&&(l=l/100*(Un(n)?n[0]:n).totalDuration()),a>1?i(e,t.substr(0,a-1),n)+l:o+l)):t==null?o:+t},Na=function(e,t,n){var s=Us(t[1]),r=(s?2:1)+(e<2?0:1),o=t[r],a,l;if(s&&(o.duration=t[1]),o.parent=n,e){for(a=o,l=n;l&&!("immediateRender"in a);)a=l.vars.defaults||{},l=ni(l.vars.inherit)&&l.parent;o.immediateRender=ni(a.immediateRender),e<2?o.runBackwards=1:o.startAt=t[r-1]}return new fn(t[0],o,t[r+1])},pr=function(e,t){return e||e===0?t(e):t},hl=function(e,t,n){return n<e?e:n>t?t:n},Fn=function(e,t){return!Sn(e)||!(t=aw.exec(e))?"":t[1]},vw=function(e,t,n){return pr(n,function(s){return hl(e,t,s)})},vd=[].slice,wx=function(e,t){return e&&cs(e)&&"length"in e&&(!t&&!e.length||e.length-1 in e&&cs(e[0]))&&!e.nodeType&&e!==Qi},Aw=function(e,t,n){return n===void 0&&(n=[]),e.forEach(function(s){var r;return Sn(s)&&!t||wx(s,1)?(r=n).push.apply(r,Pi(s)):n.push(s)})||n},Pi=function(e,t,n){return $t&&!t&&$t.selector?$t.selector(e):Sn(e)&&!n&&(md||!Jo())?vd.call((t||Th).querySelectorAll(e),0):Un(e)?Aw(e,n):wx(e)?vd.call(e,0):e?[e]:[]},Ad=function(e){return e=Pi(e)[0]||el("Invalid scope")||{},function(t){var n=e.current||e.nativeElement||e;return Pi(t,n.querySelectorAll?n:n===e?el("Invalid scope")||Th.createElement("div"):e)}},Rx=function(e){return e.sort(function(){return .5-Math.random()})},Ix=function(e){if(rn(e))return e;var t=cs(e)?e:{each:e},n=Xr(t.ease),s=t.from||0,r=parseFloat(t.base)||0,o={},a=s>0&&s<1,l=isNaN(s)||a,c=t.axis,u=s,f=s;return Sn(s)?u=f={center:.5,edges:.5,end:1}[s]||0:!a&&l&&(u=s[0],f=s[1]),function(d,h,x){var p=(x||t).length,g=o[p],m,_,v,A,S,y,M,E,b;if(!g){if(b=t.grid==="auto"?0:(t.grid||[1,Di])[1],!b){for(M=-Di;M<(M=x[b++].getBoundingClientRect().left)&&b<p;);b<p&&b--}for(g=o[p]=[],m=l?Math.min(b,p)*u-.5:s%b,_=b===Di?0:l?p*f/b-.5:s/b|0,M=0,E=Di,y=0;y<p;y++)v=y%b-m,A=_-(y/b|0),g[y]=S=c?Math.abs(c==="y"?A:v):dx(v*v+A*A),S>M&&(M=S),S<E&&(E=S);s==="random"&&Rx(g),g.max=M-E,g.min=E,g.v=p=(parseFloat(t.amount)||parseFloat(t.each)*(b>p?p-1:c?c==="y"?p/b:b:Math.max(b,p/b))||0)*(s==="edges"?-1:1),g.b=p<0?r-p:r,g.u=Fn(t.amount||t.each)||0,n=n&&p<0?kx(n):n}return p=(g[d]-g.min)/g.max||0,en(g.b+(n?n(p):p)*g.v)+g.u}},Sd=function(e){var t=Math.pow(10,((e+"").split(".")[1]||"").length);return function(n){var s=en(Math.round(parseFloat(n)/e)*e*t);return(s-s%1)/t+(Us(n)?0:Fn(n))}},Dx=function(e,t){var n=Un(e),s,r;return!n&&cs(e)&&(s=n=e.radius||Di,e.values?(e=Pi(e.values),(r=!Us(e[0]))&&(s*=s)):e=Sd(e.increment)),pr(t,n?rn(e)?function(o){return r=e(o),Math.abs(r-o)<=s?r:o}:function(o){for(var a=parseFloat(r?o.x:o),l=parseFloat(r?o.y:0),c=Di,u=0,f=e.length,d,h;f--;)r?(d=e[f].x-a,h=e[f].y-l,d=d*d+h*h):d=Math.abs(e[f]-a),d<c&&(c=d,u=f);return u=!s||c<=s?e[u]:o,r||u===o||Us(o)?u:u+Fn(o)}:Sd(e))},Px=function(e,t,n,s){return pr(Un(e)?!t:n===!0?!!(n=0):!s,function(){return Un(e)?e[~~(Math.random()*e.length)]:(n=n||1e-5)&&(s=n<1?Math.pow(10,(n+"").length-2):1)&&Math.floor(Math.round((e-n/2+Math.random()*(t-e+n*.99))/n)*n*s)/s})},Sw=function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return function(s){return t.reduce(function(r,o){return o(r)},s)}},yw=function(e,t){return function(n){return e(parseFloat(n))+(t||Fn(n))}},bw=function(e,t,n){return Lx(e,t,0,1,n)},Fx=function(e,t,n){return pr(n,function(s){return e[~~t(s)]})},Mw=function i(e,t,n){var s=t-e;return Un(e)?Fx(e,i(0,e.length),t):pr(n,function(r){return(s+(r-e)%s)%s+e})},Cw=function i(e,t,n){var s=t-e,r=s*2;return Un(e)?Fx(e,i(0,e.length-1),t):pr(n,function(o){return o=(r+(o-e)%r)%r||0,e+(o>s?r-o:o)})},nl=function(e){return e.replace(sw,function(t){var n=t.indexOf("[")+1,s=t.substring(n||7,n?t.indexOf("]"):t.length-1).split(rw);return Px(n?s:+s[0],n?0:+s[1],+s[2]||1e-5)})},Lx=function(e,t,n,s,r){var o=t-e,a=s-n;return pr(r,function(l){return n+((l-e)/o*a||0)})},Tw=function i(e,t,n,s){var r=isNaN(e+t)?0:function(h){return(1-h)*e+h*t};if(!r){var o=Sn(e),a={},l,c,u,f,d;if(n===!0&&(s=1)&&(n=null),o)e={p:e},t={p:t};else if(Un(e)&&!Un(t)){for(u=[],f=e.length,d=f-2,c=1;c<f;c++)u.push(i(e[c-1],e[c]));f--,r=function(x){x*=f;var p=Math.min(d,~~x);return u[p](x-p)},n=t}else s||(e=jo(Un(e)?[]:{},e));if(!u){for(l in t)Ph.call(a,e,l,"get",t[l]);r=function(x){return Uh(x,a)||(o?e.p:e)}}}return pr(n,r)},n0=function(e,t,n){var s=e.labels,r=Di,o,a,l;for(o in s)a=s[o]-t,a<0==!!n&&a&&r>(a=Math.abs(a))&&(l=o,r=a);return l},mi=function(e,t,n){var s=e.vars,r=s[t],o=$t,a=e._ctx,l,c,u;if(r)return l=s[t+"Params"],c=s.callbackScope||e,n&&rr.length&&Ec(),a&&($t=a),u=l?r.apply(c,l):r.call(c),$t=o,u},Sa=function(e){return ur(e),e.scrollTrigger&&e.scrollTrigger.kill(!!wn),e.progress()<1&&mi(e,"onInterrupt"),e},Co,Bx=[],Ux=function(e){if(e)if(e=!e.name&&e.default||e,Ch()||e.headless){var t=e.name,n=rn(e),s=t&&!n&&e.init?function(){this._props=[]}:e,r={init:tl,render:Uh,add:Ph,kill:Vw,modifier:Hw,rawVars:0},o={targetTest:0,get:0,getSetter:Bh,aliases:{},register:0};if(Jo(),e!==s){if(fi[t])return;Ai(s,Ai(wc(e,r),o)),jo(s.prototype,jo(r,wc(e,o))),fi[s.prop=t]=s,e.targetTest&&(uc.push(s),wh[t]=1),t=(t==="css"?"CSS":t.charAt(0).toUpperCase()+t.substr(1))+"Plugin"}_x(t,s),e.register&&e.register(oi,s,si)}else Bx.push(e)},kt=255,ya={aqua:[0,kt,kt],lime:[0,kt,0],silver:[192,192,192],black:[0,0,0],maroon:[128,0,0],teal:[0,128,128],blue:[0,0,kt],navy:[0,0,128],white:[kt,kt,kt],olive:[128,128,0],yellow:[kt,kt,0],orange:[kt,165,0],gray:[128,128,128],purple:[128,0,128],green:[0,128,0],red:[kt,0,0],pink:[kt,192,203],cyan:[0,kt,kt],transparent:[kt,kt,kt,0]},of=function(e,t,n){return e+=e<0?1:e>1?-1:0,(e*6<1?t+(n-t)*e*6:e<.5?n:e*3<2?t+(n-t)*(2/3-e)*6:t)*kt+.5|0},Ox=function(e,t,n){var s=e?Us(e)?[e>>16,e>>8&kt,e&kt]:0:ya.black,r,o,a,l,c,u,f,d,h,x;if(!s){if(e.substr(-1)===","&&(e=e.substr(0,e.length-1)),ya[e])s=ya[e];else if(e.charAt(0)==="#"){if(e.length<6&&(r=e.charAt(1),o=e.charAt(2),a=e.charAt(3),e="#"+r+r+o+o+a+a+(e.length===5?e.charAt(4)+e.charAt(4):"")),e.length===9)return s=parseInt(e.substr(1,6),16),[s>>16,s>>8&kt,s&kt,parseInt(e.substr(7),16)/255];e=parseInt(e.substr(1),16),s=[e>>16,e>>8&kt,e&kt]}else if(e.substr(0,3)==="hsl"){if(s=x=e.match($m),!t)l=+s[0]%360/360,c=+s[1]/100,u=+s[2]/100,o=u<=.5?u*(c+1):u+c-u*c,r=u*2-o,s.length>3&&(s[3]*=1),s[0]=of(l+1/3,r,o),s[1]=of(l,r,o),s[2]=of(l-1/3,r,o);else if(~e.indexOf("="))return s=e.match(px),n&&s.length<4&&(s[3]=1),s}else s=e.match($m)||ya.transparent;s=s.map(Number)}return t&&!x&&(r=s[0]/kt,o=s[1]/kt,a=s[2]/kt,f=Math.max(r,o,a),d=Math.min(r,o,a),u=(f+d)/2,f===d?l=c=0:(h=f-d,c=u>.5?h/(2-f-d):h/(f+d),l=f===r?(o-a)/h+(o<a?6:0):f===o?(a-r)/h+2:(r-o)/h+4,l*=60),s[0]=~~(l+.5),s[1]=~~(c*100+.5),s[2]=~~(u*100+.5)),n&&s.length<4&&(s[3]=1),s},Nx=function(e){var t=[],n=[],s=-1;return e.split(or).forEach(function(r){var o=r.match(Mo)||[];t.push.apply(t,o),n.push(s+=o.length+1)}),t.c=n,t},i0=function(e,t,n){var s="",r=(e+s).match(or),o=t?"hsla(":"rgba(",a=0,l,c,u,f;if(!r)return e;if(r=r.map(function(d){return(d=Ox(d,t,1))&&o+(t?d[0]+","+d[1]+"%,"+d[2]+"%,"+d[3]:d.join(","))+")"}),n&&(u=Nx(e),l=n.c,l.join(s)!==u.c.join(s)))for(c=e.replace(or,"1").split(Mo),f=c.length-1;a<f;a++)s+=c[a]+(~l.indexOf(a)?r.shift()||o+"0,0,0,0)":(u.length?u:r.length?r:n).shift());if(!c)for(c=e.split(or),f=c.length-1;a<f;a++)s+=c[a]+r[a];return s+c[f]},or=(function(){var i="(?:\\b(?:(?:rgb|rgba|hsl|hsla)\\(.+?\\))|\\B#(?:[0-9a-f]{3,4}){1,2}\\b",e;for(e in ya)i+="|"+e+"\\b";return new RegExp(i+")","gi")})(),Ew=/hsl[a]?\(/,zx=function(e){var t=e.join(" "),n;if(or.lastIndex=0,or.test(t))return n=Ew.test(t),e[1]=i0(e[1],n),e[0]=i0(e[0],n,Nx(e[1])),!0},il,di=(function(){var i=Date.now,e=500,t=33,n=i(),s=n,r=1e3/240,o=r,a=[],l,c,u,f,d,h,x=function p(g){var m=i()-s,_=g===!0,v,A,S,y;if((m>e||m<0)&&(n+=m-t),s+=m,S=s-n,v=S-o,(v>0||_)&&(y=++f.frame,d=S-f.time*1e3,f.time=S=S/1e3,o+=v+(v>=r?4:r-v),A=1),_||(l=c(p)),A)for(h=0;h<a.length;h++)a[h](S,d,y,g)};return f={time:0,frame:0,tick:function(){x(!0)},deltaRatio:function(g){return d/(1e3/(g||60))},wake:function(){gx&&(!md&&Ch()&&(Qi=md=window,Th=Qi.document||{},vi.gsap=oi,(Qi.gsapVersions||(Qi.gsapVersions=[])).push(oi.version),xx(Tc||Qi.GreenSockGlobals||!Qi.gsap&&Qi||{}),Bx.forEach(Ux)),u=typeof requestAnimationFrame<"u"&&requestAnimationFrame,l&&f.sleep(),c=u||function(g){return setTimeout(g,o-f.time*1e3+1|0)},il=1,x(2))},sleep:function(){(u?cancelAnimationFrame:clearTimeout)(l),il=0,c=tl},lagSmoothing:function(g,m){e=g||1/0,t=Math.min(m||33,e)},fps:function(g){r=1e3/(g||240),o=f.time*1e3+r},add:function(g,m,_){var v=m?function(A,S,y,M){g(A,S,y,M),f.remove(v)}:g;return f.remove(g),a[_?"unshift":"push"](v),Jo(),v},remove:function(g,m){~(m=a.indexOf(g))&&a.splice(m,1)&&h>=m&&h--},_listeners:a},f})(),Jo=function(){return!il&&di.wake()},_t={},ww=/^[\d.\-M][\d.\-,\s]/,Rw=/["']/g,Iw=function(e){for(var t={},n=e.substr(1,e.length-3).split(":"),s=n[0],r=1,o=n.length,a,l,c;r<o;r++)l=n[r],a=r!==o-1?l.lastIndexOf(","):l.length,c=l.substr(0,a),t[s]=isNaN(c)?c.replace(Rw,"").trim():+c,s=l.substr(a+1).trim();return t},Dw=function(e){var t=e.indexOf("(")+1,n=e.indexOf(")"),s=e.indexOf("(",t);return e.substring(t,~s&&s<n?e.indexOf(")",n+1):n)},Pw=function(e){var t=(e+"").split("("),n=_t[t[0]];return n&&t.length>1&&n.config?n.config.apply(null,~e.indexOf("{")?[Iw(t[1])]:Dw(e).split(",").map(yx)):_t._CE&&ww.test(e)?_t._CE("",e):n},kx=function(e){return function(t){return 1-e(1-t)}},Hx=function i(e,t){for(var n=e._first,s;n;)n instanceof qn?i(n,t):n.vars.yoyoEase&&(!n._yoyo||!n._repeat)&&n._yoyo!==t&&(n.timeline?i(n.timeline,t):(s=n._ease,n._ease=n._yEase,n._yEase=s,n._yoyo=t)),n=n._next},Xr=function(e,t){return e&&(rn(e)?e:_t[e]||Pw(e))||t},jr=function(e,t,n,s){n===void 0&&(n=function(l){return 1-t(1-l)}),s===void 0&&(s=function(l){return l<.5?t(l*2)/2:1-t((1-l)*2)/2});var r={easeIn:t,easeOut:n,easeInOut:s},o;return ii(e,function(a){_t[a]=vi[a]=r,_t[o=a.toLowerCase()]=n;for(var l in r)_t[o+(l==="easeIn"?".in":l==="easeOut"?".out":".inOut")]=_t[a+"."+l]=r[l]}),r},Vx=function(e){return function(t){return t<.5?(1-e(1-t*2))/2:.5+e((t-.5)*2)/2}},af=function i(e,t,n){var s=t>=1?t:1,r=(n||(e?.3:.45))/(t<1?t:1),o=r/pd*(Math.asin(1/s)||0),a=function(u){return u===1?1:s*Math.pow(2,-10*u)*iw((u-o)*r)+1},l=e==="out"?a:e==="in"?function(c){return 1-a(1-c)}:Vx(a);return r=pd/r,l.config=function(c,u){return i(e,c,u)},l},lf=function i(e,t){t===void 0&&(t=1.70158);var n=function(o){return o?--o*o*((t+1)*o+t)+1:0},s=e==="out"?n:e==="in"?function(r){return 1-n(1-r)}:Vx(n);return s.config=function(r){return i(e,r)},s};ii("Linear,Quad,Cubic,Quart,Quint,Strong",function(i,e){var t=e<5?e+1:e;jr(i+",Power"+(t-1),e?function(n){return Math.pow(n,t)}:function(n){return n},function(n){return 1-Math.pow(1-n,t)},function(n){return n<.5?Math.pow(n*2,t)/2:1-Math.pow((1-n)*2,t)/2})});_t.Linear.easeNone=_t.none=_t.Linear.easeIn;jr("Elastic",af("in"),af("out"),af());(function(i,e){var t=1/e,n=2*t,s=2.5*t,r=function(a){return a<t?i*a*a:a<n?i*Math.pow(a-1.5/e,2)+.75:a<s?i*(a-=2.25/e)*a+.9375:i*Math.pow(a-2.625/e,2)+.984375};jr("Bounce",function(o){return 1-r(1-o)},r)})(7.5625,2.75);jr("Expo",function(i){return Math.pow(2,10*(i-1))*i+i*i*i*i*i*i*(1-i)});jr("Circ",function(i){return-(dx(1-i*i)-1)});jr("Sine",function(i){return i===1?1:-nw(i*ew)+1});jr("Back",lf("in"),lf("out"),lf());_t.SteppedEase=_t.steps=vi.SteppedEase={config:function(e,t){e===void 0&&(e=1);var n=1/e,s=e+(t?0:1),r=t?1:0,o=1-Ht;return function(a){return((s*hl(0,o,a)|0)+r)*n}}};Ko.ease=_t["quad.out"];ii("onComplete,onUpdate,onStart,onRepeat,onReverseComplete,onInterrupt",function(i){return Rh+=i+","+i+"Params,"});var Gx=function(e,t){this.id=tw++,e._gsap=this,this.target=e,this.harness=t,this.get=t?t.get:Ax,this.set=t?t.getSetter:Bh},sl=(function(){function i(t){this.vars=t,this._delay=+t.delay||0,(this._repeat=t.repeat===1/0?-2:t.repeat||0)&&(this._rDelay=t.repeatDelay||0,this._yoyo=!!t.yoyo||!!t.yoyoEase),this._ts=1,Zo(this,+t.duration,1,1),this.data=t.data,$t&&(this._ctx=$t,$t.data.push(this)),il||di.wake()}var e=i.prototype;return e.delay=function(n){return n||n===0?(this.parent&&this.parent.smoothChildTiming&&this.startTime(this._start+n-this._delay),this._delay=n,this):this._delay},e.duration=function(n){return arguments.length?this.totalDuration(this._repeat>0?n+(n+this._rDelay)*this._repeat:n):this.totalDuration()&&this._dur},e.totalDuration=function(n){return arguments.length?(this._dirty=0,Zo(this,this._repeat<0?n:(n-this._repeat*this._rDelay)/(this._repeat+1))):this._tDur},e.totalTime=function(n,s){if(Jo(),!arguments.length)return this._tTime;var r=this._dp;if(r&&r.smoothChildTiming&&this._ts){for($c(this,n),!r._dp||r.parent||Cx(r,this);r&&r.parent;)r.parent._time!==r._start+(r._ts>=0?r._tTime/r._ts:(r.totalDuration()-r._tTime)/-r._ts)&&r.totalTime(r._tTime,!0),r=r.parent;!this.parent&&this._dp.autoRemoveChildren&&(this._ts>0&&n<this._tDur||this._ts<0&&n>0||!this._tDur&&!n)&&Zi(this._dp,this,this._start-this._delay)}return(this._tTime!==n||!this._dur&&!s||this._initted&&Math.abs(this._zTime)===Ht||!this._initted&&this._dur&&n||!n&&!this._initted&&(this.add||this._ptLookup))&&(this._ts||(this._pTime=n),Sx(this,n,s)),this},e.time=function(n,s){return arguments.length?this.totalTime(Math.min(this.totalDuration(),n+e0(this))%(this._dur+this._rDelay)||(n?this._dur:0),s):this._time},e.totalProgress=function(n,s){return arguments.length?this.totalTime(this.totalDuration()*n,s):this.totalDuration()?Math.min(1,this._tTime/this._tDur):this.rawTime()>=0&&this._initted?1:0},e.progress=function(n,s){return arguments.length?this.totalTime(this.duration()*(this._yoyo&&!(this.iteration()&1)?1-n:n)+e0(this),s):this.duration()?Math.min(1,this._time/this._dur):this.rawTime()>0?1:0},e.iteration=function(n,s){var r=this.duration()+this._rDelay;return arguments.length?this.totalTime(this._time+(n-1)*r,s):this._repeat?$o(this._tTime,r)+1:1},e.timeScale=function(n,s){if(!arguments.length)return this._rts===-Ht?0:this._rts;if(this._rts===n)return this;var r=this.parent&&this._ts?Rc(this.parent._time,this):this._tTime;return this._rts=+n||0,this._ts=this._ps||n===-Ht?0:this._rts,this.totalTime(hl(-Math.abs(this._delay),this.totalDuration(),r),s!==!1),jc(this),hw(this)},e.paused=function(n){return arguments.length?(this._ps!==n&&(this._ps=n,n?(this._pTime=this._tTime||Math.max(-this._delay,this.rawTime()),this._ts=this._act=0):(Jo(),this._ts=this._rts,this.totalTime(this.parent&&!this.parent.smoothChildTiming?this.rawTime():this._tTime||this._pTime,this.progress()===1&&Math.abs(this._zTime)!==Ht&&(this._tTime-=Ht)))),this):this._ps},e.startTime=function(n){if(arguments.length){this._start=en(n);var s=this.parent||this._dp;return s&&(s._sort||!this.parent)&&Zi(s,this,this._start-this._delay),this}return this._start},e.endTime=function(n){return this._start+(ni(n)?this.totalDuration():this.duration())/Math.abs(this._ts||1)},e.rawTime=function(n){var s=this.parent||this._dp;return s?n&&(!this._ts||this._repeat&&this._time&&this.totalProgress()<1)?this._tTime%(this._dur+this._rDelay):this._ts?Rc(s.rawTime(n),this):this._tTime:this._tTime},e.revert=function(n){n===void 0&&(n=cw);var s=wn;return wn=n,Dh(this)&&(this.timeline&&this.timeline.revert(n),this.totalTime(-.01,n.suppressEvents)),this.data!=="nested"&&n.kill!==!1&&this.kill(),wn=s,this},e.globalTime=function(n){for(var s=this,r=arguments.length?n:s.rawTime();s;)r=s._start+r/(Math.abs(s._ts)||1),s=s._dp;return!this.parent&&this._sat?this._sat.globalTime(n):r},e.repeat=function(n){return arguments.length?(this._repeat=n===1/0?-2:n,t0(this)):this._repeat===-2?1/0:this._repeat},e.repeatDelay=function(n){if(arguments.length){var s=this._time;return this._rDelay=n,t0(this),s?this.time(s):this}return this._rDelay},e.yoyo=function(n){return arguments.length?(this._yoyo=n,this):this._yoyo},e.seek=function(n,s){return this.totalTime(Mi(this,n),ni(s))},e.restart=function(n,s){return this.play().totalTime(n?-this._delay:0,ni(s)),this._dur||(this._zTime=-Ht),this},e.play=function(n,s){return n!=null&&this.seek(n,s),this.reversed(!1).paused(!1)},e.reverse=function(n,s){return n!=null&&this.seek(n||this.totalDuration(),s),this.reversed(!0).paused(!1)},e.pause=function(n,s){return n!=null&&this.seek(n,s),this.paused(!0)},e.resume=function(){return this.paused(!1)},e.reversed=function(n){return arguments.length?(!!n!==this.reversed()&&this.timeScale(-this._rts||(n?-Ht:0)),this):this._rts<0},e.invalidate=function(){return this._initted=this._act=0,this._zTime=-Ht,this},e.isActive=function(){var n=this.parent||this._dp,s=this._start,r;return!!(!n||this._ts&&this._initted&&n.isActive()&&(r=n.rawTime(!0))>=s&&r<this.endTime(!0)-Ht)},e.eventCallback=function(n,s,r){var o=this.vars;return arguments.length>1?(s?(o[n]=s,r&&(o[n+"Params"]=r),n==="onUpdate"&&(this._onUpdate=s)):delete o[n],this):o[n]},e.then=function(n){var s=this,r=s._prom;return new Promise(function(o){var a=rn(n)?n:bx,l=function(){var u=s.then;s.then=null,r&&r(),rn(a)&&(a=a(s))&&(a.then||a===s)&&(s.then=u),o(a),s.then=u};s._initted&&s.totalProgress()===1&&s._ts>=0||!s._tTime&&s._ts<0?l():s._prom=l})},e.kill=function(){Sa(this)},i})();Ai(sl.prototype,{_time:0,_start:0,_end:0,_tTime:0,_tDur:0,_dirty:0,_repeat:0,_yoyo:!1,parent:null,_initted:!1,_rDelay:0,_ts:1,_dp:0,ratio:0,_zTime:-Ht,_prom:0,_ps:!1,_rts:1});var qn=(function(i){fx(e,i);function e(n,s){var r;return n===void 0&&(n={}),r=i.call(this,n)||this,r.labels={},r.smoothChildTiming=!!n.smoothChildTiming,r.autoRemoveChildren=!!n.autoRemoveChildren,r._sort=ni(n.sortChildren),tn&&Zi(n.parent||tn,bs(r),s),n.reversed&&r.reverse(),n.paused&&r.paused(!0),n.scrollTrigger&&Tx(bs(r),n.scrollTrigger),r}var t=e.prototype;return t.to=function(s,r,o){return Na(0,arguments,this),this},t.from=function(s,r,o){return Na(1,arguments,this),this},t.fromTo=function(s,r,o,a){return Na(2,arguments,this),this},t.set=function(s,r,o){return r.duration=0,r.parent=this,Oa(r).repeatDelay||(r.repeat=0),r.immediateRender=!!r.immediateRender,new fn(s,r,Mi(this,o),1),this},t.call=function(s,r,o){return Zi(this,fn.delayedCall(0,s,r),o)},t.staggerTo=function(s,r,o,a,l,c,u){return o.duration=r,o.stagger=o.stagger||a,o.onComplete=c,o.onCompleteParams=u,o.parent=this,new fn(s,o,Mi(this,l)),this},t.staggerFrom=function(s,r,o,a,l,c,u){return o.runBackwards=1,Oa(o).immediateRender=ni(o.immediateRender),this.staggerTo(s,r,o,a,l,c,u)},t.staggerFromTo=function(s,r,o,a,l,c,u,f){return a.startAt=o,Oa(a).immediateRender=ni(a.immediateRender),this.staggerTo(s,r,a,l,c,u,f)},t.render=function(s,r,o){var a=this._time,l=this._dirty?this.totalDuration():this._tDur,c=this._dur,u=s<=0?0:en(s),f=this._zTime<0!=s<0&&(this._initted||!c),d,h,x,p,g,m,_,v,A,S,y,M;if(this!==tn&&u>l&&s>=0&&(u=l),u!==this._tTime||o||f){if(a!==this._time&&c&&(u+=this._time-a,s+=this._time-a),d=u,A=this._start,v=this._ts,m=!v,f&&(c||(a=this._zTime),(s||!r)&&(this._zTime=s)),this._repeat){if(y=this._yoyo,g=c+this._rDelay,this._repeat<-1&&s<0)return this.totalTime(g*100+s,r,o);if(d=en(u%g),u===l?(p=this._repeat,d=c):(S=en(u/g),p=~~S,p&&p===S&&(d=c,p--),d>c&&(d=c)),S=$o(this._tTime,g),!a&&this._tTime&&S!==p&&this._tTime-S*g-this._dur<=0&&(S=p),y&&p&1&&(d=c-d,M=1),p!==S&&!this._lock){var E=y&&S&1,b=E===(y&&p&1);if(p<S&&(E=!E),a=E?0:u%c?c:u,this._lock=1,this.render(a||(M?0:en(p*g)),r,!c)._lock=0,this._tTime=u,!r&&this.parent&&mi(this,"onRepeat"),this.vars.repeatRefresh&&!M&&(this.invalidate()._lock=1,S=p),a&&a!==this._time||m!==!this._ts||this.vars.onRepeat&&!this.parent&&!this._act)return this;if(c=this._dur,l=this._tDur,b&&(this._lock=2,a=E?c:-1e-4,this.render(a,!0),this.vars.repeatRefresh&&!M&&this.invalidate()),this._lock=0,!this._ts&&!m)return this;Hx(this,M)}}if(this._hasPause&&!this._forcing&&this._lock<2&&(_=xw(this,en(a),en(d)),_&&(u-=d-(d=_._start))),this._tTime=u,this._time=d,this._act=!v,this._initted||(this._onUpdate=this.vars.onUpdate,this._initted=1,this._zTime=s,a=0),!a&&u&&c&&!r&&!S&&(mi(this,"onStart"),this._tTime!==u))return this;if(d>=a&&s>=0)for(h=this._first;h;){if(x=h._next,(h._act||d>=h._start)&&h._ts&&_!==h){if(h.parent!==this)return this.render(s,r,o);if(h.render(h._ts>0?(d-h._start)*h._ts:(h._dirty?h.totalDuration():h._tDur)+(d-h._start)*h._ts,r,o),d!==this._time||!this._ts&&!m){_=0,x&&(u+=this._zTime=-Ht);break}}h=x}else{h=this._last;for(var C=s<0?s:d;h;){if(x=h._prev,(h._act||C<=h._end)&&h._ts&&_!==h){if(h.parent!==this)return this.render(s,r,o);if(h.render(h._ts>0?(C-h._start)*h._ts:(h._dirty?h.totalDuration():h._tDur)+(C-h._start)*h._ts,r,o||wn&&Dh(h)),d!==this._time||!this._ts&&!m){_=0,x&&(u+=this._zTime=C?-Ht:Ht);break}}h=x}}if(_&&!r&&(this.pause(),_.render(d>=a?0:-Ht)._zTime=d>=a?1:-1,this._ts))return this._start=A,jc(this),this.render(s,r,o);this._onUpdate&&!r&&mi(this,"onUpdate",!0),(u===l&&this._tTime>=this.totalDuration()||!u&&a)&&(A===this._start||Math.abs(v)!==Math.abs(this._ts))&&(this._lock||((s||!c)&&(u===l&&this._ts>0||!u&&this._ts<0)&&ur(this,1),!r&&!(s<0&&!a)&&(u||a||!l)&&(mi(this,u===l&&s>=0?"onComplete":"onReverseComplete",!0),this._prom&&!(u<l&&this.timeScale()>0)&&this._prom())))}return this},t.add=function(s,r){var o=this;if(Us(r)||(r=Mi(this,r,s)),!(s instanceof sl)){if(Un(s))return s.forEach(function(a){return o.add(a,r)}),this;if(Sn(s))return this.addLabel(s,r);if(rn(s))s=fn.delayedCall(0,s);else return this}return this!==s?Zi(this,s,r):this},t.getChildren=function(s,r,o,a){s===void 0&&(s=!0),r===void 0&&(r=!0),o===void 0&&(o=!0),a===void 0&&(a=-Di);for(var l=[],c=this._first;c;)c._start>=a&&(c instanceof fn?r&&l.push(c):(o&&l.push(c),s&&l.push.apply(l,c.getChildren(!0,r,o)))),c=c._next;return l},t.getById=function(s){for(var r=this.getChildren(1,1,1),o=r.length;o--;)if(r[o].vars.id===s)return r[o]},t.remove=function(s){return Sn(s)?this.removeLabel(s):rn(s)?this.killTweensOf(s):(s.parent===this&&Kc(this,s),s===this._recent&&(this._recent=this._last),Wr(this))},t.totalTime=function(s,r){return arguments.length?(this._forcing=1,!this._dp&&this._ts&&(this._start=en(di.time-(this._ts>0?s/this._ts:(this.totalDuration()-s)/-this._ts))),i.prototype.totalTime.call(this,s,r),this._forcing=0,this):this._tTime},t.addLabel=function(s,r){return this.labels[s]=Mi(this,r),this},t.removeLabel=function(s){return delete this.labels[s],this},t.addPause=function(s,r,o){var a=fn.delayedCall(0,r||tl,o);return a.data="isPause",this._hasPause=1,Zi(this,a,Mi(this,s))},t.removePause=function(s){var r=this._first;for(s=Mi(this,s);r;)r._start===s&&r.data==="isPause"&&ur(r),r=r._next},t.killTweensOf=function(s,r,o){for(var a=this.getTweensOf(s,o),l=a.length;l--;)Js!==a[l]&&a[l].kill(s,r);return this},t.getTweensOf=function(s,r){for(var o=[],a=Pi(s),l=this._first,c=Us(r),u;l;)l instanceof fn?uw(l._targets,a)&&(c?(!Js||l._initted&&l._ts)&&l.globalTime(0)<=r&&l.globalTime(l.totalDuration())>r:!r||l.isActive())&&o.push(l):(u=l.getTweensOf(a,r)).length&&o.push.apply(o,u),l=l._next;return o},t.tweenTo=function(s,r){r=r||{};var o=this,a=Mi(o,s),l=r,c=l.startAt,u=l.onStart,f=l.onStartParams,d=l.immediateRender,h,x=fn.to(o,Ai({ease:r.ease||"none",lazy:!1,immediateRender:!1,time:a,overwrite:"auto",duration:r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale())||Ht,onStart:function(){if(o.pause(),!h){var g=r.duration||Math.abs((a-(c&&"time"in c?c.time:o._time))/o.timeScale());x._dur!==g&&Zo(x,g,0,1).render(x._time,!0,!0),h=1}u&&u.apply(x,f||[])}},r));return d?x.render(0):x},t.tweenFromTo=function(s,r,o){return this.tweenTo(r,Ai({startAt:{time:Mi(this,s)}},o))},t.recent=function(){return this._recent},t.nextLabel=function(s){return s===void 0&&(s=this._time),n0(this,Mi(this,s))},t.previousLabel=function(s){return s===void 0&&(s=this._time),n0(this,Mi(this,s),1)},t.currentLabel=function(s){return arguments.length?this.seek(s,!0):this.previousLabel(this._time+Ht)},t.shiftChildren=function(s,r,o){o===void 0&&(o=0);var a=this._first,l=this.labels,c;for(s=en(s);a;)a._start>=o&&(a._start+=s,a._end+=s),a=a._next;if(r)for(c in l)l[c]>=o&&(l[c]+=s);return Wr(this)},t.invalidate=function(s){var r=this._first;for(this._lock=0;r;)r.invalidate(s),r=r._next;return i.prototype.invalidate.call(this,s)},t.clear=function(s){s===void 0&&(s=!0);for(var r=this._first,o;r;)o=r._next,this.remove(r),r=o;return this._dp&&(this._time=this._tTime=this._pTime=0),s&&(this.labels={}),Wr(this)},t.totalDuration=function(s){var r=0,o=this,a=o._last,l=Di,c,u,f;if(arguments.length)return o.timeScale((o._repeat<0?o.duration():o.totalDuration())/(o.reversed()?-s:s));if(o._dirty){for(f=o.parent;a;)c=a._prev,a._dirty&&a.totalDuration(),u=a._start,u>l&&o._sort&&a._ts&&!o._lock?(o._lock=1,Zi(o,a,u-a._delay,1)._lock=0):l=u,u<0&&a._ts&&(r-=u,(!f&&!o._dp||f&&f.smoothChildTiming)&&(o._start+=en(u/o._ts),o._time-=u,o._tTime-=u),o.shiftChildren(-u,!1,-1/0),l=0),a._end>r&&a._ts&&(r=a._end),a=c;Zo(o,o===tn&&o._time>r?o._time:r,1,1),o._dirty=0}return o._tDur},e.updateRoot=function(s){if(tn._ts&&(Sx(tn,Rc(s,tn)),vx=di.frame),di.frame>=Zm){Zm+=_i.autoSleep||120;var r=tn._first;if((!r||!r._ts)&&_i.autoSleep&&di._listeners.length<2){for(;r&&!r._ts;)r=r._next;r||di.sleep()}}},e})(sl);Ai(qn.prototype,{_lock:0,_hasPause:0,_forcing:0});var Fw=function(e,t,n,s,r,o,a){var l=new si(this._pt,e,t,0,1,Kx,null,r),c=0,u=0,f,d,h,x,p,g,m,_;for(l.b=n,l.e=s,n+="",s+="",(m=~s.indexOf("random("))&&(s=nl(s)),o&&(_=[n,s],o(_,e,t),n=_[0],s=_[1]),d=n.match(sf)||[];f=sf.exec(s);)x=f[0],p=s.substring(c,f.index),h?h=(h+1)%5:p.substr(-5)==="rgba("&&(h=1),x!==d[u++]&&(g=parseFloat(d[u-1])||0,l._pt={_next:l._pt,p:p||u===1?p:",",s:g,c:x.charAt(1)==="="?Uo(g,x)-g:parseFloat(x)-g,m:h&&h<4?Math.round:0},c=sf.lastIndex);return l.c=c<s.length?s.substring(c,s.length):"",l.fp=a,(mx.test(s)||m)&&(l.e=0),this._pt=l,l},Ph=function(e,t,n,s,r,o,a,l,c,u){rn(s)&&(s=s(r||0,e,o));var f=e[t],d=n!=="get"?n:rn(f)?c?e[t.indexOf("set")||!rn(e["get"+t.substr(3)])?t:"get"+t.substr(3)](c):e[t]():f,h=rn(f)?c?Nw:Yx:Lh,x;if(Sn(s)&&(~s.indexOf("random(")&&(s=nl(s)),s.charAt(1)==="="&&(x=Uo(d,s)+(Fn(d)||0),(x||x===0)&&(s=x))),!u||d!==s||yd)return!isNaN(d*s)&&s!==""?(x=new si(this._pt,e,t,+d||0,s-(d||0),typeof f=="boolean"?kw:Qx,0,h),c&&(x.fp=c),a&&x.modifier(a,this,e),this._pt=x):(!f&&!(t in e)&&Eh(t,s),Fw.call(this,e,t,d,s,h,l||_i.stringFilter,c))},Lw=function(e,t,n,s,r){if(rn(e)&&(e=za(e,r,t,n,s)),!cs(e)||e.style&&e.nodeType||Un(e)||hx(e))return Sn(e)?za(e,r,t,n,s):e;var o={},a;for(a in e)o[a]=za(e[a],r,t,n,s);return o},Wx=function(e,t,n,s,r,o){var a,l,c,u;if(fi[e]&&(a=new fi[e]).init(r,a.rawVars?t[e]:Lw(t[e],s,r,o,n),n,s,o)!==!1&&(n._pt=l=new si(n._pt,r,e,0,1,a.render,a,0,a.priority),n!==Co))for(c=n._ptLookup[n._targets.indexOf(r)],u=a._props.length;u--;)c[a._props[u]]=l;return a},Js,yd,Fh=function i(e,t,n){var s=e.vars,r=s.ease,o=s.startAt,a=s.immediateRender,l=s.lazy,c=s.onUpdate,u=s.runBackwards,f=s.yoyoEase,d=s.keyframes,h=s.autoRevert,x=e._dur,p=e._startAt,g=e._targets,m=e.parent,_=m&&m.data==="nested"?m.vars.targets:g,v=e._overwrite==="auto"&&!bh,A=e.timeline,S,y,M,E,b,C,P,B,N,k,V,F,q;if(A&&(!d||!r)&&(r="none"),e._ease=Xr(r,Ko.ease),e._yEase=f?kx(Xr(f===!0?r:f,Ko.ease)):0,f&&e._yoyo&&!e._repeat&&(f=e._yEase,e._yEase=e._ease,e._ease=f),e._from=!A&&!!s.runBackwards,!A||d&&!s.stagger){if(B=g[0]?Gr(g[0]).harness:0,F=B&&s[B.prop],S=wc(s,wh),p&&(p._zTime<0&&p.progress(1),t<0&&u&&a&&!h?p.render(-1,!0):p.revert(u&&x?cc:lw),p._lazy=0),o){if(ur(e._startAt=fn.set(g,Ai({data:"isStart",overwrite:!1,parent:m,immediateRender:!0,lazy:!p&&ni(l),startAt:null,delay:0,onUpdate:c&&function(){return mi(e,"onUpdate")},stagger:0},o))),e._startAt._dp=0,e._startAt._sat=e,t<0&&(wn||!a&&!h)&&e._startAt.revert(cc),a&&x&&t<=0&&n<=0){t&&(e._zTime=t);return}}else if(u&&x&&!p){if(t&&(a=!1),M=Ai({overwrite:!1,data:"isFromStart",lazy:a&&!p&&ni(l),immediateRender:a,stagger:0,parent:m},S),F&&(M[B.prop]=F),ur(e._startAt=fn.set(g,M)),e._startAt._dp=0,e._startAt._sat=e,t<0&&(wn?e._startAt.revert(cc):e._startAt.render(-1,!0)),e._zTime=t,!a)i(e._startAt,Ht,Ht);else if(!t)return}for(e._pt=e._ptCache=0,l=x&&ni(l)||l&&!x,y=0;y<g.length;y++){if(b=g[y],P=b._gsap||Ih(g)[y]._gsap,e._ptLookup[y]=k={},gd[P.id]&&rr.length&&Ec(),V=_===g?y:_.indexOf(b),B&&(N=new B).init(b,F||S,e,V,_)!==!1&&(e._pt=E=new si(e._pt,b,N.name,0,1,N.render,N,0,N.priority),N._props.forEach(function(G){k[G]=E}),N.priority&&(C=1)),!B||F)for(M in S)fi[M]&&(N=Wx(M,S,e,V,b,_))?N.priority&&(C=1):k[M]=E=Ph.call(e,b,M,"get",S[M],V,_,0,s.stringFilter);e._op&&e._op[y]&&e.kill(b,e._op[y]),v&&e._pt&&(Js=e,tn.killTweensOf(b,k,e.globalTime(t)),q=!e.parent,Js=0),e._pt&&l&&(gd[P.id]=1)}C&&jx(e),e._onInit&&e._onInit(e)}e._onUpdate=c,e._initted=(!e._op||e._pt)&&!q,d&&t<=0&&A.render(Di,!0,!0)},Bw=function(e,t,n,s,r,o,a,l){var c=(e._pt&&e._ptCache||(e._ptCache={}))[t],u,f,d,h;if(!c)for(c=e._ptCache[t]=[],d=e._ptLookup,h=e._targets.length;h--;){if(u=d[h][t],u&&u.d&&u.d._pt)for(u=u.d._pt;u&&u.p!==t&&u.fp!==t;)u=u._next;if(!u)return yd=1,e.vars[t]="+=0",Fh(e,a),yd=0,l?el(t+" not eligible for reset"):1;c.push(u)}for(h=c.length;h--;)f=c[h],u=f._pt||f,u.s=(s||s===0)&&!r?s:u.s+(s||0)+o*u.c,u.c=n-u.s,f.e&&(f.e=on(n)+Fn(f.e)),f.b&&(f.b=u.s+Fn(f.b))},Uw=function(e,t){var n=e[0]?Gr(e[0]).harness:0,s=n&&n.aliases,r,o,a,l;if(!s)return t;r=jo({},t);for(o in s)if(o in r)for(l=s[o].split(","),a=l.length;a--;)r[l[a]]=r[o];return r},Ow=function(e,t,n,s){var r=t.ease||s||"power1.inOut",o,a;if(Un(t))a=n[e]||(n[e]=[]),t.forEach(function(l,c){return a.push({t:c/(t.length-1)*100,v:l,e:r})});else for(o in t)a=n[o]||(n[o]=[]),o==="ease"||a.push({t:parseFloat(e),v:t[o],e:r})},za=function(e,t,n,s,r){return rn(e)?e.call(t,n,s,r):Sn(e)&&~e.indexOf("random(")?nl(e):e},Xx=Rh+"repeat,repeatDelay,yoyo,repeatRefresh,yoyoEase,autoRevert",qx={};ii(Xx+",id,stagger,delay,duration,paused,scrollTrigger",function(i){return qx[i]=1});var fn=(function(i){fx(e,i);function e(n,s,r,o){var a;typeof s=="number"&&(r.duration=s,s=r,r=null),a=i.call(this,o?s:Oa(s))||this;var l=a.vars,c=l.duration,u=l.delay,f=l.immediateRender,d=l.stagger,h=l.overwrite,x=l.keyframes,p=l.defaults,g=l.scrollTrigger,m=l.yoyoEase,_=s.parent||tn,v=(Un(n)||hx(n)?Us(n[0]):"length"in s)?[n]:Pi(n),A,S,y,M,E,b,C,P;if(a._targets=v.length?Ih(v):el("GSAP target "+n+" not found. https://gsap.com",!_i.nullTargetWarn)||[],a._ptLookup=[],a._overwrite=h,x||d||Zl(c)||Zl(u)){if(s=a.vars,A=a.timeline=new qn({data:"nested",defaults:p||{},targets:_&&_.data==="nested"?_.vars.targets:v}),A.kill(),A.parent=A._dp=bs(a),A._start=0,d||Zl(c)||Zl(u)){if(M=v.length,C=d&&Ix(d),cs(d))for(E in d)~Xx.indexOf(E)&&(P||(P={}),P[E]=d[E]);for(S=0;S<M;S++)y=wc(s,qx),y.stagger=0,m&&(y.yoyoEase=m),P&&jo(y,P),b=v[S],y.duration=+za(c,bs(a),S,b,v),y.delay=(+za(u,bs(a),S,b,v)||0)-a._delay,!d&&M===1&&y.delay&&(a._delay=u=y.delay,a._start+=u,y.delay=0),A.to(b,y,C?C(S,b,v):0),A._ease=_t.none;A.duration()?c=u=0:a.timeline=0}else if(x){Oa(Ai(A.vars.defaults,{ease:"none"})),A._ease=Xr(x.ease||s.ease||"none");var B=0,N,k,V;if(Un(x))x.forEach(function(F){return A.to(v,F,">")}),A.duration();else{y={};for(E in x)E==="ease"||E==="easeEach"||Ow(E,x[E],y,x.easeEach);for(E in y)for(N=y[E].sort(function(F,q){return F.t-q.t}),B=0,S=0;S<N.length;S++)k=N[S],V={ease:k.e,duration:(k.t-(S?N[S-1].t:0))/100*c},V[E]=k.v,A.to(v,V,B),B+=V.duration;A.duration()<c&&A.to({},{duration:c-A.duration()})}}c||a.duration(c=A.duration())}else a.timeline=0;return h===!0&&!bh&&(Js=bs(a),tn.killTweensOf(v),Js=0),Zi(_,bs(a),r),s.reversed&&a.reverse(),s.paused&&a.paused(!0),(f||!c&&!x&&a._start===en(_._time)&&ni(f)&&pw(bs(a))&&_.data!=="nested")&&(a._tTime=-Ht,a.render(Math.max(0,-u)||0)),g&&Tx(bs(a),g),a}var t=e.prototype;return t.render=function(s,r,o){var a=this._time,l=this._tDur,c=this._dur,u=s<0,f=s>l-Ht&&!u?l:s<Ht?0:s,d,h,x,p,g,m,_,v,A;if(!c)gw(this,s,r,o);else if(f!==this._tTime||!s||o||!this._initted&&this._tTime||this._startAt&&this._zTime<0!==u||this._lazy){if(d=f,v=this.timeline,this._repeat){if(p=c+this._rDelay,this._repeat<-1&&u)return this.totalTime(p*100+s,r,o);if(d=en(f%p),f===l?(x=this._repeat,d=c):(g=en(f/p),x=~~g,x&&x===g?(d=c,x--):d>c&&(d=c)),m=this._yoyo&&x&1,m&&(A=this._yEase,d=c-d),g=$o(this._tTime,p),d===a&&!o&&this._initted&&x===g)return this._tTime=f,this;x!==g&&(v&&this._yEase&&Hx(v,m),this.vars.repeatRefresh&&!m&&!this._lock&&d!==p&&this._initted&&(this._lock=o=1,this.render(en(p*x),!0).invalidate()._lock=0))}if(!this._initted){if(Ex(this,u?s:d,o,r,f))return this._tTime=0,this;if(a!==this._time&&!(o&&this.vars.repeatRefresh&&x!==g))return this;if(c!==this._dur)return this.render(s,r,o)}if(this._tTime=f,this._time=d,!this._act&&this._ts&&(this._act=1,this._lazy=0),this.ratio=_=(A||this._ease)(d/c),this._from&&(this.ratio=_=1-_),!a&&f&&!r&&!g&&(mi(this,"onStart"),this._tTime!==f))return this;for(h=this._pt;h;)h.r(_,h.d),h=h._next;v&&v.render(s<0?s:v._dur*v._ease(d/this._dur),r,o)||this._startAt&&(this._zTime=s),this._onUpdate&&!r&&(u&&xd(this,s,r,o),mi(this,"onUpdate")),this._repeat&&x!==g&&this.vars.onRepeat&&!r&&this.parent&&mi(this,"onRepeat"),(f===this._tDur||!f)&&this._tTime===f&&(u&&!this._onUpdate&&xd(this,s,!0,!0),(s||!c)&&(f===this._tDur&&this._ts>0||!f&&this._ts<0)&&ur(this,1),!r&&!(u&&!a)&&(f||a||m)&&(mi(this,f===l?"onComplete":"onReverseComplete",!0),this._prom&&!(f<l&&this.timeScale()>0)&&this._prom()))}return this},t.targets=function(){return this._targets},t.invalidate=function(s){return(!s||!this.vars.runBackwards)&&(this._startAt=0),this._pt=this._op=this._onUpdate=this._lazy=this.ratio=0,this._ptLookup=[],this.timeline&&this.timeline.invalidate(s),i.prototype.invalidate.call(this,s)},t.resetTo=function(s,r,o,a,l){il||di.wake(),this._ts||this.play();var c=Math.min(this._dur,(this._dp._time-this._start)*this._ts),u;return this._initted||Fh(this,c),u=this._ease(c/this._dur),Bw(this,s,r,o,a,u,c,l)?this.resetTo(s,r,o,a,1):($c(this,0),this.parent||Mx(this._dp,this,"_first","_last",this._dp._sort?"_start":0),this.render(0))},t.kill=function(s,r){if(r===void 0&&(r="all"),!s&&(!r||r==="all"))return this._lazy=this._pt=0,this.parent?Sa(this):this.scrollTrigger&&this.scrollTrigger.kill(!!wn),this;if(this.timeline){var o=this.timeline.totalDuration();return this.timeline.killTweensOf(s,r,Js&&Js.vars.overwrite!==!0)._first||Sa(this),this.parent&&o!==this.timeline.totalDuration()&&Zo(this,this._dur*this.timeline._tDur/o,0,1),this}var a=this._targets,l=s?Pi(s):a,c=this._ptLookup,u=this._pt,f,d,h,x,p,g,m;if((!r||r==="all")&&dw(a,l))return r==="all"&&(this._pt=0),Sa(this);for(f=this._op=this._op||[],r!=="all"&&(Sn(r)&&(p={},ii(r,function(_){return p[_]=1}),r=p),r=Uw(a,r)),m=a.length;m--;)if(~l.indexOf(a[m])){d=c[m],r==="all"?(f[m]=r,x=d,h={}):(h=f[m]=f[m]||{},x=r);for(p in x)g=d&&d[p],g&&((!("kill"in g.d)||g.d.kill(p)===!0)&&Kc(this,g,"_pt"),delete d[p]),h!=="all"&&(h[p]=1)}return this._initted&&!this._pt&&u&&Sa(this),this},e.to=function(s,r){return new e(s,r,arguments[2])},e.from=function(s,r){return Na(1,arguments)},e.delayedCall=function(s,r,o,a){return new e(r,0,{immediateRender:!1,lazy:!1,overwrite:!1,delay:s,onComplete:r,onReverseComplete:r,onCompleteParams:o,onReverseCompleteParams:o,callbackScope:a})},e.fromTo=function(s,r,o){return Na(2,arguments)},e.set=function(s,r){return r.duration=0,r.repeatDelay||(r.repeat=0),new e(s,r)},e.killTweensOf=function(s,r,o){return tn.killTweensOf(s,r,o)},e})(sl);Ai(fn.prototype,{_targets:[],_lazy:0,_startAt:0,_op:0,_onInit:0});ii("staggerTo,staggerFrom,staggerFromTo",function(i){fn[i]=function(){var e=new qn,t=vd.call(arguments,0);return t.splice(i==="staggerFromTo"?5:4,0,0),e[i].apply(e,t)}});var Lh=function(e,t,n){return e[t]=n},Yx=function(e,t,n){return e[t](n)},Nw=function(e,t,n,s){return e[t](s.fp,n)},zw=function(e,t,n){return e.setAttribute(t,n)},Bh=function(e,t){return rn(e[t])?Yx:Mh(e[t])&&e.setAttribute?zw:Lh},Qx=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e6)/1e6,t)},kw=function(e,t){return t.set(t.t,t.p,!!(t.s+t.c*e),t)},Kx=function(e,t){var n=t._pt,s="";if(!e&&t.b)s=t.b;else if(e===1&&t.e)s=t.e;else{for(;n;)s=n.p+(n.m?n.m(n.s+n.c*e):Math.round((n.s+n.c*e)*1e4)/1e4)+s,n=n._next;s+=t.c}t.set(t.t,t.p,s,t)},Uh=function(e,t){for(var n=t._pt;n;)n.r(e,n.d),n=n._next},Hw=function(e,t,n,s){for(var r=this._pt,o;r;)o=r._next,r.p===s&&r.modifier(e,t,n),r=o},Vw=function(e){for(var t=this._pt,n,s;t;)s=t._next,t.p===e&&!t.op||t.op===e?Kc(this,t,"_pt"):t.dep||(n=1),t=s;return!n},Gw=function(e,t,n,s){s.mSet(e,t,s.m.call(s.tween,n,s.mt),s)},jx=function(e){for(var t=e._pt,n,s,r,o;t;){for(n=t._next,s=r;s&&s.pr>t.pr;)s=s._next;(t._prev=s?s._prev:o)?t._prev._next=t:r=t,(t._next=s)?s._prev=t:o=t,t=n}e._pt=r},si=(function(){function i(t,n,s,r,o,a,l,c,u){this.t=n,this.s=r,this.c=o,this.p=s,this.r=a||Qx,this.d=l||this,this.set=c||Lh,this.pr=u||0,this._next=t,t&&(t._prev=this)}var e=i.prototype;return e.modifier=function(n,s,r){this.mSet=this.mSet||this.set,this.set=Gw,this.m=n,this.mt=r,this.tween=s},i})();ii(Rh+"parent,duration,ease,delay,overwrite,runBackwards,startAt,yoyo,immediateRender,repeat,repeatDelay,data,paused,reversed,lazy,callbackScope,stringFilter,id,yoyoEase,stagger,inherit,repeatRefresh,keyframes,autoRevert,scrollTrigger",function(i){return wh[i]=1});vi.TweenMax=vi.TweenLite=fn;vi.TimelineLite=vi.TimelineMax=qn;tn=new qn({sortChildren:!1,defaults:Ko,autoRemoveChildren:!0,id:"root",smoothChildTiming:!0});_i.stringFilter=zx;var qr=[],fc={},Ww=[],s0=0,Xw=0,cf=function(e){return(fc[e]||Ww).map(function(t){return t()})},bd=function(){var e=Date.now(),t=[];e-s0>2&&(cf("matchMediaInit"),qr.forEach(function(n){var s=n.queries,r=n.conditions,o,a,l,c;for(a in s)o=Qi.matchMedia(s[a]).matches,o&&(l=1),o!==r[a]&&(r[a]=o,c=1);c&&(n.revert(),l&&t.push(n))}),cf("matchMediaRevert"),t.forEach(function(n){return n.onMatch(n,function(s){return n.add(null,s)})}),s0=e,cf("matchMedia"))},$x=(function(){function i(t,n){this.selector=n&&Ad(n),this.data=[],this._r=[],this.isReverted=!1,this.id=Xw++,t&&this.add(t)}var e=i.prototype;return e.add=function(n,s,r){rn(n)&&(r=s,s=n,n=rn);var o=this,a=function(){var c=$t,u=o.selector,f;return c&&c!==o&&c.data.push(o),r&&(o.selector=Ad(r)),$t=o,f=s.apply(o,arguments),rn(f)&&o._r.push(f),$t=c,o.selector=u,o.isReverted=!1,f};return o.last=a,n===rn?a(o,function(l){return o.add(null,l)}):n?o[n]=a:a},e.ignore=function(n){var s=$t;$t=null,n(this),$t=s},e.getTweens=function(){var n=[];return this.data.forEach(function(s){return s instanceof i?n.push.apply(n,s.getTweens()):s instanceof fn&&!(s.parent&&s.parent.data==="nested")&&n.push(s)}),n},e.clear=function(){this._r.length=this.data.length=0},e.kill=function(n,s){var r=this;if(n?(function(){for(var a=r.getTweens(),l=r.data.length,c;l--;)c=r.data[l],c.data==="isFlip"&&(c.revert(),c.getChildren(!0,!0,!1).forEach(function(u){return a.splice(a.indexOf(u),1)}));for(a.map(function(u){return{g:u._dur||u._delay||u._sat&&!u._sat.vars.immediateRender?u.globalTime(0):-1/0,t:u}}).sort(function(u,f){return f.g-u.g||-1/0}).forEach(function(u){return u.t.revert(n)}),l=r.data.length;l--;)c=r.data[l],c instanceof qn?c.data!=="nested"&&(c.scrollTrigger&&c.scrollTrigger.revert(),c.kill()):!(c instanceof fn)&&c.revert&&c.revert(n);r._r.forEach(function(u){return u(n,r)}),r.isReverted=!0})():this.data.forEach(function(a){return a.kill&&a.kill()}),this.clear(),s)for(var o=qr.length;o--;)qr[o].id===this.id&&qr.splice(o,1)},e.revert=function(n){this.kill(n||{})},i})(),qw=(function(){function i(t){this.contexts=[],this.scope=t,$t&&$t.data.push(this)}var e=i.prototype;return e.add=function(n,s,r){cs(n)||(n={matches:n});var o=new $x(0,r||this.scope),a=o.conditions={},l,c,u;$t&&!o.selector&&(o.selector=$t.selector),this.contexts.push(o),s=o.add("onMatch",s),o.queries=n;for(c in n)c==="all"?u=1:(l=Qi.matchMedia(n[c]),l&&(qr.indexOf(o)<0&&qr.push(o),(a[c]=l.matches)&&(u=1),l.addListener?l.addListener(bd):l.addEventListener("change",bd)));return u&&s(o,function(f){return o.add(null,f)}),this},e.revert=function(n){this.kill(n||{})},e.kill=function(n){this.contexts.forEach(function(s){return s.kill(n,!0)})},i})(),Ic={registerPlugin:function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];t.forEach(function(s){return Ux(s)})},timeline:function(e){return new qn(e)},getTweensOf:function(e,t){return tn.getTweensOf(e,t)},getProperty:function(e,t,n,s){Sn(e)&&(e=Pi(e)[0]);var r=Gr(e||{}).get,o=n?bx:yx;return n==="native"&&(n=""),e&&(t?o((fi[t]&&fi[t].get||r)(e,t,n,s)):function(a,l,c){return o((fi[a]&&fi[a].get||r)(e,a,l,c))})},quickSetter:function(e,t,n){if(e=Pi(e),e.length>1){var s=e.map(function(u){return oi.quickSetter(u,t,n)}),r=s.length;return function(u){for(var f=r;f--;)s[f](u)}}e=e[0]||{};var o=fi[t],a=Gr(e),l=a.harness&&(a.harness.aliases||{})[t]||t,c=o?function(u){var f=new o;Co._pt=0,f.init(e,n?u+n:u,Co,0,[e]),f.render(1,f),Co._pt&&Uh(1,Co)}:a.set(e,l);return o?c:function(u){return c(e,l,n?u+n:u,a,1)}},quickTo:function(e,t,n){var s,r=oi.to(e,Ai((s={},s[t]="+=0.1",s.paused=!0,s.stagger=0,s),n||{})),o=function(l,c,u){return r.resetTo(t,l,c,u)};return o.tween=r,o},isTweening:function(e){return tn.getTweensOf(e,!0).length>0},defaults:function(e){return e&&e.ease&&(e.ease=Xr(e.ease,Ko.ease)),Jm(Ko,e||{})},config:function(e){return Jm(_i,e||{})},registerEffect:function(e){var t=e.name,n=e.effect,s=e.plugins,r=e.defaults,o=e.extendTimeline;(s||"").split(",").forEach(function(a){return a&&!fi[a]&&!vi[a]&&el(t+" effect requires "+a+" plugin.")}),rf[t]=function(a,l,c){return n(Pi(a),Ai(l||{},r),c)},o&&(qn.prototype[t]=function(a,l,c){return this.add(rf[t](a,cs(l)?l:(c=l)&&{},this),c)})},registerEase:function(e,t){_t[e]=Xr(t)},parseEase:function(e,t){return arguments.length?Xr(e,t):_t},getById:function(e){return tn.getById(e)},exportRoot:function(e,t){e===void 0&&(e={});var n=new qn(e),s,r;for(n.smoothChildTiming=ni(e.smoothChildTiming),tn.remove(n),n._dp=0,n._time=n._tTime=tn._time,s=tn._first;s;)r=s._next,(t||!(!s._dur&&s instanceof fn&&s.vars.onComplete===s._targets[0]))&&Zi(n,s,s._start-s._delay),s=r;return Zi(tn,n,0),n},context:function(e,t){return e?new $x(e,t):$t},matchMedia:function(e){return new qw(e)},matchMediaRefresh:function(){return qr.forEach(function(e){var t=e.conditions,n,s;for(s in t)t[s]&&(t[s]=!1,n=1);n&&e.revert()})||bd()},addEventListener:function(e,t){var n=fc[e]||(fc[e]=[]);~n.indexOf(t)||n.push(t)},removeEventListener:function(e,t){var n=fc[e],s=n&&n.indexOf(t);s>=0&&n.splice(s,1)},utils:{wrap:Mw,wrapYoyo:Cw,distribute:Ix,random:Px,snap:Dx,normalize:bw,getUnit:Fn,clamp:vw,splitColor:Ox,toArray:Pi,selector:Ad,mapRange:Lx,pipe:Sw,unitize:yw,interpolate:Tw,shuffle:Rx},install:xx,effects:rf,ticker:di,updateRoot:qn.updateRoot,plugins:fi,globalTimeline:tn,core:{PropTween:si,globals:_x,Tween:fn,Timeline:qn,Animation:sl,getCache:Gr,_removeLinkedListItem:Kc,reverting:function(){return wn},context:function(e){return e&&$t&&($t.data.push(e),e._ctx=$t),$t},suppressOverwrites:function(e){return bh=e}}};ii("to,from,fromTo,delayedCall,set,killTweensOf",function(i){return Ic[i]=fn[i]});di.add(qn.updateRoot);Co=Ic.to({},{duration:0});var Yw=function(e,t){for(var n=e._pt;n&&n.p!==t&&n.op!==t&&n.fp!==t;)n=n._next;return n},Qw=function(e,t){var n=e._targets,s,r,o;for(s in t)for(r=n.length;r--;)o=e._ptLookup[r][s],o&&(o=o.d)&&(o._pt&&(o=Yw(o,s)),o&&o.modifier&&o.modifier(t[s],e,n[r],s))},uf=function(e,t){return{name:e,headless:1,rawVars:1,init:function(s,r,o){o._onInit=function(a){var l,c;if(Sn(r)&&(l={},ii(r,function(u){return l[u]=1}),r=l),t){l={};for(c in r)l[c]=t(r[c]);r=l}Qw(a,r)}}}},oi=Ic.registerPlugin({name:"attr",init:function(e,t,n,s,r){var o,a,l;this.tween=n;for(o in t)l=e.getAttribute(o)||"",a=this.add(e,"setAttribute",(l||0)+"",t[o],s,r,0,0,o),a.op=o,a.b=l,this._props.push(o)},render:function(e,t){for(var n=t._pt;n;)wn?n.set(n.t,n.p,n.b,n):n.r(e,n.d),n=n._next}},{name:"endArray",headless:1,init:function(e,t){for(var n=t.length;n--;)this.add(e,n,e[n]||0,t[n],0,0,0,0,0,1)}},uf("roundProps",Sd),uf("modifiers"),uf("snap",Dx))||Ic;fn.version=qn.version=oi.version="3.14.2";gx=1;Ch()&&Jo();_t.Power0;_t.Power1;_t.Power2;_t.Power3;_t.Power4;_t.Linear;_t.Quad;_t.Cubic;_t.Quart;_t.Quint;_t.Strong;_t.Elastic;_t.Back;_t.SteppedEase;_t.Bounce;_t.Sine;_t.Expo;_t.Circ;var r0,er,Oo,Oh,kr,o0,Nh,Kw=function(){return typeof window<"u"},Os={},Pr=180/Math.PI,No=Math.PI/180,vo=Math.atan2,a0=1e8,zh=/([A-Z])/g,jw=/(left|right|width|margin|padding|x)/i,$w=/[\s,\(]\S/,ns={autoAlpha:"opacity,visibility",scale:"scaleX,scaleY",alpha:"opacity"},Md=function(e,t){return t.set(t.t,t.p,Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},Zw=function(e,t){return t.set(t.t,t.p,e===1?t.e:Math.round((t.s+t.c*e)*1e4)/1e4+t.u,t)},Jw=function(e,t){return t.set(t.t,t.p,e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},e3=function(e,t){return t.set(t.t,t.p,e===1?t.e:e?Math.round((t.s+t.c*e)*1e4)/1e4+t.u:t.b,t)},t3=function(e,t){var n=t.s+t.c*e;t.set(t.t,t.p,~~(n+(n<0?-.5:.5))+t.u,t)},Zx=function(e,t){return t.set(t.t,t.p,e?t.e:t.b,t)},Jx=function(e,t){return t.set(t.t,t.p,e!==1?t.b:t.e,t)},n3=function(e,t,n){return e.style[t]=n},i3=function(e,t,n){return e.style.setProperty(t,n)},s3=function(e,t,n){return e._gsap[t]=n},r3=function(e,t,n){return e._gsap.scaleX=e._gsap.scaleY=n},o3=function(e,t,n,s,r){var o=e._gsap;o.scaleX=o.scaleY=n,o.renderTransform(r,o)},a3=function(e,t,n,s,r){var o=e._gsap;o[t]=n,o.renderTransform(r,o)},nn="transform",ri=nn+"Origin",l3=function i(e,t){var n=this,s=this.target,r=s.style,o=s._gsap;if(e in Os&&r){if(this.tfm=this.tfm||{},e!=="transform")e=ns[e]||e,~e.indexOf(",")?e.split(",").forEach(function(a){return n.tfm[a]=Ts(s,a)}):this.tfm[e]=o.x?o[e]:Ts(s,e),e===ri&&(this.tfm.zOrigin=o.zOrigin);else return ns.transform.split(",").forEach(function(a){return i.call(n,a,t)});if(this.props.indexOf(nn)>=0)return;o.svg&&(this.svgo=s.getAttribute("data-svg-origin"),this.props.push(ri,t,"")),e=nn}(r||t)&&this.props.push(e,t,r[e])},e_=function(e){e.translate&&(e.removeProperty("translate"),e.removeProperty("scale"),e.removeProperty("rotate"))},c3=function(){var e=this.props,t=this.target,n=t.style,s=t._gsap,r,o;for(r=0;r<e.length;r+=3)e[r+1]?e[r+1]===2?t[e[r]](e[r+2]):t[e[r]]=e[r+2]:e[r+2]?n[e[r]]=e[r+2]:n.removeProperty(e[r].substr(0,2)==="--"?e[r]:e[r].replace(zh,"-$1").toLowerCase());if(this.tfm){for(o in this.tfm)s[o]=this.tfm[o];s.svg&&(s.renderTransform(),t.setAttribute("data-svg-origin",this.svgo||"")),r=Nh(),(!r||!r.isStart)&&!n[nn]&&(e_(n),s.zOrigin&&n[ri]&&(n[ri]+=" "+s.zOrigin+"px",s.zOrigin=0,s.renderTransform()),s.uncache=1)}},t_=function(e,t){var n={target:e,props:[],revert:c3,save:l3};return e._gsap||oi.core.getCache(e),t&&e.style&&e.nodeType&&t.split(",").forEach(function(s){return n.save(s)}),n},n_,Cd=function(e,t){var n=er.createElementNS?er.createElementNS((t||"http://www.w3.org/1999/xhtml").replace(/^https/,"http"),e):er.createElement(e);return n&&n.style?n:er.createElement(e)},gi=function i(e,t,n){var s=getComputedStyle(e);return s[t]||s.getPropertyValue(t.replace(zh,"-$1").toLowerCase())||s.getPropertyValue(t)||!n&&i(e,ea(t)||t,1)||""},l0="O,Moz,ms,Ms,Webkit".split(","),ea=function(e,t,n){var s=t||kr,r=s.style,o=5;if(e in r&&!n)return e;for(e=e.charAt(0).toUpperCase()+e.substr(1);o--&&!(l0[o]+e in r););return o<0?null:(o===3?"ms":o>=0?l0[o]:"")+e},Td=function(){Kw()&&window.document&&(r0=window,er=r0.document,Oo=er.documentElement,kr=Cd("div")||{style:{}},Cd("div"),nn=ea(nn),ri=nn+"Origin",kr.style.cssText="border-width:0;line-height:0;position:absolute;padding:0",n_=!!ea("perspective"),Nh=oi.core.reverting,Oh=1)},c0=function(e){var t=e.ownerSVGElement,n=Cd("svg",t&&t.getAttribute("xmlns")||"http://www.w3.org/2000/svg"),s=e.cloneNode(!0),r;s.style.display="block",n.appendChild(s),Oo.appendChild(n);try{r=s.getBBox()}catch{}return n.removeChild(s),Oo.removeChild(n),r},u0=function(e,t){for(var n=t.length;n--;)if(e.hasAttribute(t[n]))return e.getAttribute(t[n])},i_=function(e){var t,n;try{t=e.getBBox()}catch{t=c0(e),n=1}return t&&(t.width||t.height)||n||(t=c0(e)),t&&!t.width&&!t.x&&!t.y?{x:+u0(e,["x","cx","x1"])||0,y:+u0(e,["y","cy","y1"])||0,width:0,height:0}:t},s_=function(e){return!!(e.getCTM&&(!e.parentNode||e.ownerSVGElement)&&i_(e))},fr=function(e,t){if(t){var n=e.style,s;t in Os&&t!==ri&&(t=nn),n.removeProperty?(s=t.substr(0,2),(s==="ms"||t.substr(0,6)==="webkit")&&(t="-"+t),n.removeProperty(s==="--"?t:t.replace(zh,"-$1").toLowerCase())):n.removeAttribute(t)}},tr=function(e,t,n,s,r,o){var a=new si(e._pt,t,n,0,1,o?Jx:Zx);return e._pt=a,a.b=s,a.e=r,e._props.push(n),a},f0={deg:1,rad:1,turn:1},u3={grid:1,flex:1},dr=function i(e,t,n,s){var r=parseFloat(n)||0,o=(n+"").trim().substr((r+"").length)||"px",a=kr.style,l=jw.test(t),c=e.tagName.toLowerCase()==="svg",u=(c?"client":"offset")+(l?"Width":"Height"),f=100,d=s==="px",h=s==="%",x,p,g,m;if(s===o||!r||f0[s]||f0[o])return r;if(o!=="px"&&!d&&(r=i(e,t,n,"px")),m=e.getCTM&&s_(e),(h||o==="%")&&(Os[t]||~t.indexOf("adius")))return x=m?e.getBBox()[l?"width":"height"]:e[u],on(h?r/x*f:r/100*x);if(a[l?"width":"height"]=f+(d?o:s),p=s!=="rem"&&~t.indexOf("adius")||s==="em"&&e.appendChild&&!c?e:e.parentNode,m&&(p=(e.ownerSVGElement||{}).parentNode),(!p||p===er||!p.appendChild)&&(p=er.body),g=p._gsap,g&&h&&g.width&&l&&g.time===di.time&&!g.uncache)return on(r/g.width*f);if(h&&(t==="height"||t==="width")){var _=e.style[t];e.style[t]=f+s,x=e[u],_?e.style[t]=_:fr(e,t)}else(h||o==="%")&&!u3[gi(p,"display")]&&(a.position=gi(e,"position")),p===e&&(a.position="static"),p.appendChild(kr),x=kr[u],p.removeChild(kr),a.position="absolute";return l&&h&&(g=Gr(p),g.time=di.time,g.width=p[u]),on(d?x*r/f:x&&r?f/x*r:0)},Ts=function(e,t,n,s){var r;return Oh||Td(),t in ns&&t!=="transform"&&(t=ns[t],~t.indexOf(",")&&(t=t.split(",")[0])),Os[t]&&t!=="transform"?(r=ol(e,s),r=t!=="transformOrigin"?r[t]:r.svg?r.origin:Pc(gi(e,ri))+" "+r.zOrigin+"px"):(r=e.style[t],(!r||r==="auto"||s||~(r+"").indexOf("calc("))&&(r=Dc[t]&&Dc[t](e,t,n)||gi(e,t)||Ax(e,t)||(t==="opacity"?1:0))),n&&!~(r+"").trim().indexOf(" ")?dr(e,t,r,n)+n:r},f3=function(e,t,n,s){if(!n||n==="none"){var r=ea(t,e,1),o=r&&gi(e,r,1);o&&o!==n?(t=r,n=o):t==="borderColor"&&(n=gi(e,"borderTopColor"))}var a=new si(this._pt,e.style,t,0,1,Kx),l=0,c=0,u,f,d,h,x,p,g,m,_,v,A,S;if(a.b=n,a.e=s,n+="",s+="",s.substring(0,6)==="var(--"&&(s=gi(e,s.substring(4,s.indexOf(")")))),s==="auto"&&(p=e.style[t],e.style[t]=s,s=gi(e,t)||s,p?e.style[t]=p:fr(e,t)),u=[n,s],zx(u),n=u[0],s=u[1],d=n.match(Mo)||[],S=s.match(Mo)||[],S.length){for(;f=Mo.exec(s);)g=f[0],_=s.substring(l,f.index),x?x=(x+1)%5:(_.substr(-5)==="rgba("||_.substr(-5)==="hsla(")&&(x=1),g!==(p=d[c++]||"")&&(h=parseFloat(p)||0,A=p.substr((h+"").length),g.charAt(1)==="="&&(g=Uo(h,g)+A),m=parseFloat(g),v=g.substr((m+"").length),l=Mo.lastIndex-v.length,v||(v=v||_i.units[t]||A,l===s.length&&(s+=v,a.e+=v)),A!==v&&(h=dr(e,t,p,v)||0),a._pt={_next:a._pt,p:_||c===1?_:",",s:h,c:m-h,m:x&&x<4||t==="zIndex"?Math.round:0});a.c=l<s.length?s.substring(l,s.length):""}else a.r=t==="display"&&s==="none"?Jx:Zx;return mx.test(s)&&(a.e=0),this._pt=a,a},d0={top:"0%",bottom:"100%",left:"0%",right:"100%",center:"50%"},d3=function(e){var t=e.split(" "),n=t[0],s=t[1]||"50%";return(n==="top"||n==="bottom"||s==="left"||s==="right")&&(e=n,n=s,s=e),t[0]=d0[n]||n,t[1]=d0[s]||s,t.join(" ")},h3=function(e,t){if(t.tween&&t.tween._time===t.tween._dur){var n=t.t,s=n.style,r=t.u,o=n._gsap,a,l,c;if(r==="all"||r===!0)s.cssText="",l=1;else for(r=r.split(","),c=r.length;--c>-1;)a=r[c],Os[a]&&(l=1,a=a==="transformOrigin"?ri:nn),fr(n,a);l&&(fr(n,nn),o&&(o.svg&&n.removeAttribute("transform"),s.scale=s.rotate=s.translate="none",ol(n,1),o.uncache=1,e_(s)))}},Dc={clearProps:function(e,t,n,s,r){if(r.data!=="isFromStart"){var o=e._pt=new si(e._pt,t,n,0,0,h3);return o.u=s,o.pr=-10,o.tween=r,e._props.push(n),1}}},rl=[1,0,0,1,0,0],r_={},o_=function(e){return e==="matrix(1, 0, 0, 1, 0, 0)"||e==="none"||!e},h0=function(e){var t=gi(e,nn);return o_(t)?rl:t.substr(7).match(px).map(on)},kh=function(e,t){var n=e._gsap||Gr(e),s=e.style,r=h0(e),o,a,l,c;return n.svg&&e.getAttribute("transform")?(l=e.transform.baseVal.consolidate().matrix,r=[l.a,l.b,l.c,l.d,l.e,l.f],r.join(",")==="1,0,0,1,0,0"?rl:r):(r===rl&&!e.offsetParent&&e!==Oo&&!n.svg&&(l=s.display,s.display="block",o=e.parentNode,(!o||!e.offsetParent&&!e.getBoundingClientRect().width)&&(c=1,a=e.nextElementSibling,Oo.appendChild(e)),r=h0(e),l?s.display=l:fr(e,"display"),c&&(a?o.insertBefore(e,a):o?o.appendChild(e):Oo.removeChild(e))),t&&r.length>6?[r[0],r[1],r[4],r[5],r[12],r[13]]:r)},Ed=function(e,t,n,s,r,o){var a=e._gsap,l=r||kh(e,!0),c=a.xOrigin||0,u=a.yOrigin||0,f=a.xOffset||0,d=a.yOffset||0,h=l[0],x=l[1],p=l[2],g=l[3],m=l[4],_=l[5],v=t.split(" "),A=parseFloat(v[0])||0,S=parseFloat(v[1])||0,y,M,E,b;n?l!==rl&&(M=h*g-x*p)&&(E=A*(g/M)+S*(-p/M)+(p*_-g*m)/M,b=A*(-x/M)+S*(h/M)-(h*_-x*m)/M,A=E,S=b):(y=i_(e),A=y.x+(~v[0].indexOf("%")?A/100*y.width:A),S=y.y+(~(v[1]||v[0]).indexOf("%")?S/100*y.height:S)),s||s!==!1&&a.smooth?(m=A-c,_=S-u,a.xOffset=f+(m*h+_*p)-m,a.yOffset=d+(m*x+_*g)-_):a.xOffset=a.yOffset=0,a.xOrigin=A,a.yOrigin=S,a.smooth=!!s,a.origin=t,a.originIsAbsolute=!!n,e.style[ri]="0px 0px",o&&(tr(o,a,"xOrigin",c,A),tr(o,a,"yOrigin",u,S),tr(o,a,"xOffset",f,a.xOffset),tr(o,a,"yOffset",d,a.yOffset)),e.setAttribute("data-svg-origin",A+" "+S)},ol=function(e,t){var n=e._gsap||new Gx(e);if("x"in n&&!t&&!n.uncache)return n;var s=e.style,r=n.scaleX<0,o="px",a="deg",l=getComputedStyle(e),c=gi(e,ri)||"0",u,f,d,h,x,p,g,m,_,v,A,S,y,M,E,b,C,P,B,N,k,V,F,q,G,j,ce,ye,Ie,Ne,de,qe;return u=f=d=p=g=m=_=v=A=0,h=x=1,n.svg=!!(e.getCTM&&s_(e)),l.translate&&((l.translate!=="none"||l.scale!=="none"||l.rotate!=="none")&&(s[nn]=(l.translate!=="none"?"translate3d("+(l.translate+" 0 0").split(" ").slice(0,3).join(", ")+") ":"")+(l.rotate!=="none"?"rotate("+l.rotate+") ":"")+(l.scale!=="none"?"scale("+l.scale.split(" ").join(",")+") ":"")+(l[nn]!=="none"?l[nn]:"")),s.scale=s.rotate=s.translate="none"),M=kh(e,n.svg),n.svg&&(n.uncache?(G=e.getBBox(),c=n.xOrigin-G.x+"px "+(n.yOrigin-G.y)+"px",q=""):q=!t&&e.getAttribute("data-svg-origin"),Ed(e,q||c,!!q||n.originIsAbsolute,n.smooth!==!1,M)),S=n.xOrigin||0,y=n.yOrigin||0,M!==rl&&(P=M[0],B=M[1],N=M[2],k=M[3],u=V=M[4],f=F=M[5],M.length===6?(h=Math.sqrt(P*P+B*B),x=Math.sqrt(k*k+N*N),p=P||B?vo(B,P)*Pr:0,_=N||k?vo(N,k)*Pr+p:0,_&&(x*=Math.abs(Math.cos(_*No))),n.svg&&(u-=S-(S*P+y*N),f-=y-(S*B+y*k))):(qe=M[6],Ne=M[7],ce=M[8],ye=M[9],Ie=M[10],de=M[11],u=M[12],f=M[13],d=M[14],E=vo(qe,Ie),g=E*Pr,E&&(b=Math.cos(-E),C=Math.sin(-E),q=V*b+ce*C,G=F*b+ye*C,j=qe*b+Ie*C,ce=V*-C+ce*b,ye=F*-C+ye*b,Ie=qe*-C+Ie*b,de=Ne*-C+de*b,V=q,F=G,qe=j),E=vo(-N,Ie),m=E*Pr,E&&(b=Math.cos(-E),C=Math.sin(-E),q=P*b-ce*C,G=B*b-ye*C,j=N*b-Ie*C,de=k*C+de*b,P=q,B=G,N=j),E=vo(B,P),p=E*Pr,E&&(b=Math.cos(E),C=Math.sin(E),q=P*b+B*C,G=V*b+F*C,B=B*b-P*C,F=F*b-V*C,P=q,V=G),g&&Math.abs(g)+Math.abs(p)>359.9&&(g=p=0,m=180-m),h=on(Math.sqrt(P*P+B*B+N*N)),x=on(Math.sqrt(F*F+qe*qe)),E=vo(V,F),_=Math.abs(E)>2e-4?E*Pr:0,A=de?1/(de<0?-de:de):0),n.svg&&(q=e.getAttribute("transform"),n.forceCSS=e.setAttribute("transform","")||!o_(gi(e,nn)),q&&e.setAttribute("transform",q))),Math.abs(_)>90&&Math.abs(_)<270&&(r?(h*=-1,_+=p<=0?180:-180,p+=p<=0?180:-180):(x*=-1,_+=_<=0?180:-180)),t=t||n.uncache,n.x=u-((n.xPercent=u&&(!t&&n.xPercent||(Math.round(e.offsetWidth/2)===Math.round(-u)?-50:0)))?e.offsetWidth*n.xPercent/100:0)+o,n.y=f-((n.yPercent=f&&(!t&&n.yPercent||(Math.round(e.offsetHeight/2)===Math.round(-f)?-50:0)))?e.offsetHeight*n.yPercent/100:0)+o,n.z=d+o,n.scaleX=on(h),n.scaleY=on(x),n.rotation=on(p)+a,n.rotationX=on(g)+a,n.rotationY=on(m)+a,n.skewX=_+a,n.skewY=v+a,n.transformPerspective=A+o,(n.zOrigin=parseFloat(c.split(" ")[2])||!t&&n.zOrigin||0)&&(s[ri]=Pc(c)),n.xOffset=n.yOffset=0,n.force3D=_i.force3D,n.renderTransform=n.svg?m3:n_?a_:p3,n.uncache=0,n},Pc=function(e){return(e=e.split(" "))[0]+" "+e[1]},ff=function(e,t,n){var s=Fn(t);return on(parseFloat(t)+parseFloat(dr(e,"x",n+"px",s)))+s},p3=function(e,t){t.z="0px",t.rotationY=t.rotationX="0deg",t.force3D=0,a_(e,t)},Tr="0deg",xa="0px",Er=") ",a_=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.z,c=n.rotation,u=n.rotationY,f=n.rotationX,d=n.skewX,h=n.skewY,x=n.scaleX,p=n.scaleY,g=n.transformPerspective,m=n.force3D,_=n.target,v=n.zOrigin,A="",S=m==="auto"&&e&&e!==1||m===!0;if(v&&(f!==Tr||u!==Tr)){var y=parseFloat(u)*No,M=Math.sin(y),E=Math.cos(y),b;y=parseFloat(f)*No,b=Math.cos(y),o=ff(_,o,M*b*-v),a=ff(_,a,-Math.sin(y)*-v),l=ff(_,l,E*b*-v+v)}g!==xa&&(A+="perspective("+g+Er),(s||r)&&(A+="translate("+s+"%, "+r+"%) "),(S||o!==xa||a!==xa||l!==xa)&&(A+=l!==xa||S?"translate3d("+o+", "+a+", "+l+") ":"translate("+o+", "+a+Er),c!==Tr&&(A+="rotate("+c+Er),u!==Tr&&(A+="rotateY("+u+Er),f!==Tr&&(A+="rotateX("+f+Er),(d!==Tr||h!==Tr)&&(A+="skew("+d+", "+h+Er),(x!==1||p!==1)&&(A+="scale("+x+", "+p+Er),_.style[nn]=A||"translate(0, 0)"},m3=function(e,t){var n=t||this,s=n.xPercent,r=n.yPercent,o=n.x,a=n.y,l=n.rotation,c=n.skewX,u=n.skewY,f=n.scaleX,d=n.scaleY,h=n.target,x=n.xOrigin,p=n.yOrigin,g=n.xOffset,m=n.yOffset,_=n.forceCSS,v=parseFloat(o),A=parseFloat(a),S,y,M,E,b;l=parseFloat(l),c=parseFloat(c),u=parseFloat(u),u&&(u=parseFloat(u),c+=u,l+=u),l||c?(l*=No,c*=No,S=Math.cos(l)*f,y=Math.sin(l)*f,M=Math.sin(l-c)*-d,E=Math.cos(l-c)*d,c&&(u*=No,b=Math.tan(c-u),b=Math.sqrt(1+b*b),M*=b,E*=b,u&&(b=Math.tan(u),b=Math.sqrt(1+b*b),S*=b,y*=b)),S=on(S),y=on(y),M=on(M),E=on(E)):(S=f,E=d,y=M=0),(v&&!~(o+"").indexOf("px")||A&&!~(a+"").indexOf("px"))&&(v=dr(h,"x",o,"px"),A=dr(h,"y",a,"px")),(x||p||g||m)&&(v=on(v+x-(x*S+p*M)+g),A=on(A+p-(x*y+p*E)+m)),(s||r)&&(b=h.getBBox(),v=on(v+s/100*b.width),A=on(A+r/100*b.height)),b="matrix("+S+","+y+","+M+","+E+","+v+","+A+")",h.setAttribute("transform",b),_&&(h.style[nn]=b)},g3=function(e,t,n,s,r){var o=360,a=Sn(r),l=parseFloat(r)*(a&&~r.indexOf("rad")?Pr:1),c=l-s,u=s+c+"deg",f,d;return a&&(f=r.split("_")[1],f==="short"&&(c%=o,c!==c%(o/2)&&(c+=c<0?o:-o)),f==="cw"&&c<0?c=(c+o*a0)%o-~~(c/o)*o:f==="ccw"&&c>0&&(c=(c-o*a0)%o-~~(c/o)*o)),e._pt=d=new si(e._pt,t,n,s,c,Zw),d.e=u,d.u="deg",e._props.push(n),d},p0=function(e,t){for(var n in t)e[n]=t[n];return e},x3=function(e,t,n){var s=p0({},n._gsap),r="perspective,force3D,transformOrigin,svgOrigin",o=n.style,a,l,c,u,f,d,h,x;s.svg?(c=n.getAttribute("transform"),n.setAttribute("transform",""),o[nn]=t,a=ol(n,1),fr(n,nn),n.setAttribute("transform",c)):(c=getComputedStyle(n)[nn],o[nn]=t,a=ol(n,1),o[nn]=c);for(l in Os)c=s[l],u=a[l],c!==u&&r.indexOf(l)<0&&(h=Fn(c),x=Fn(u),f=h!==x?dr(n,l,c,x):parseFloat(c),d=parseFloat(u),e._pt=new si(e._pt,a,l,f,d-f,Md),e._pt.u=x||0,e._props.push(l));p0(a,s)};ii("padding,margin,Width,Radius",function(i,e){var t="Top",n="Right",s="Bottom",r="Left",o=(e<3?[t,n,s,r]:[t+r,t+n,s+n,s+r]).map(function(a){return e<2?i+a:"border"+a+i});Dc[e>1?"border"+i:i]=function(a,l,c,u,f){var d,h;if(arguments.length<4)return d=o.map(function(x){return Ts(a,x,c)}),h=d.join(" "),h.split(d[0]).length===5?d[0]:h;d=(u+"").split(" "),h={},o.forEach(function(x,p){return h[x]=d[p]=d[p]||d[(p-1)/2|0]}),a.init(l,h,f)}});var l_={name:"css",register:Td,targetTest:function(e){return e.style&&e.nodeType},init:function(e,t,n,s,r){var o=this._props,a=e.style,l=n.vars.startAt,c,u,f,d,h,x,p,g,m,_,v,A,S,y,M,E,b;Oh||Td(),this.styles=this.styles||t_(e),E=this.styles.props,this.tween=n;for(p in t)if(p!=="autoRound"&&(u=t[p],!(fi[p]&&Wx(p,t,n,s,e,r)))){if(h=typeof u,x=Dc[p],h==="function"&&(u=u.call(n,s,e,r),h=typeof u),h==="string"&&~u.indexOf("random(")&&(u=nl(u)),x)x(this,e,p,u,n)&&(M=1);else if(p.substr(0,2)==="--")c=(getComputedStyle(e).getPropertyValue(p)+"").trim(),u+="",or.lastIndex=0,or.test(c)||(g=Fn(c),m=Fn(u),m?g!==m&&(c=dr(e,p,c,m)+m):g&&(u+=g)),this.add(a,"setProperty",c,u,s,r,0,0,p),o.push(p),E.push(p,0,a[p]);else if(h!=="undefined"){if(l&&p in l?(c=typeof l[p]=="function"?l[p].call(n,s,e,r):l[p],Sn(c)&&~c.indexOf("random(")&&(c=nl(c)),Fn(c+"")||c==="auto"||(c+=_i.units[p]||Fn(Ts(e,p))||""),(c+"").charAt(1)==="="&&(c=Ts(e,p))):c=Ts(e,p),d=parseFloat(c),_=h==="string"&&u.charAt(1)==="="&&u.substr(0,2),_&&(u=u.substr(2)),f=parseFloat(u),p in ns&&(p==="autoAlpha"&&(d===1&&Ts(e,"visibility")==="hidden"&&f&&(d=0),E.push("visibility",0,a.visibility),tr(this,a,"visibility",d?"inherit":"hidden",f?"inherit":"hidden",!f)),p!=="scale"&&p!=="transform"&&(p=ns[p],~p.indexOf(",")&&(p=p.split(",")[0]))),v=p in Os,v){if(this.styles.save(p),b=u,h==="string"&&u.substring(0,6)==="var(--"){if(u=gi(e,u.substring(4,u.indexOf(")"))),u.substring(0,5)==="calc("){var C=e.style.perspective;e.style.perspective=u,u=gi(e,"perspective"),C?e.style.perspective=C:fr(e,"perspective")}f=parseFloat(u)}if(A||(S=e._gsap,S.renderTransform&&!t.parseTransform||ol(e,t.parseTransform),y=t.smoothOrigin!==!1&&S.smooth,A=this._pt=new si(this._pt,a,nn,0,1,S.renderTransform,S,0,-1),A.dep=1),p==="scale")this._pt=new si(this._pt,S,"scaleY",S.scaleY,(_?Uo(S.scaleY,_+f):f)-S.scaleY||0,Md),this._pt.u=0,o.push("scaleY",p),p+="X";else if(p==="transformOrigin"){E.push(ri,0,a[ri]),u=d3(u),S.svg?Ed(e,u,0,y,0,this):(m=parseFloat(u.split(" ")[2])||0,m!==S.zOrigin&&tr(this,S,"zOrigin",S.zOrigin,m),tr(this,a,p,Pc(c),Pc(u)));continue}else if(p==="svgOrigin"){Ed(e,u,1,y,0,this);continue}else if(p in r_){g3(this,S,p,d,_?Uo(d,_+u):u);continue}else if(p==="smoothOrigin"){tr(this,S,"smooth",S.smooth,u);continue}else if(p==="force3D"){S[p]=u;continue}else if(p==="transform"){x3(this,u,e);continue}}else p in a||(p=ea(p)||p);if(v||(f||f===0)&&(d||d===0)&&!$w.test(u)&&p in a)g=(c+"").substr((d+"").length),f||(f=0),m=Fn(u)||(p in _i.units?_i.units[p]:g),g!==m&&(d=dr(e,p,c,m)),this._pt=new si(this._pt,v?S:a,p,d,(_?Uo(d,_+f):f)-d,!v&&(m==="px"||p==="zIndex")&&t.autoRound!==!1?t3:Md),this._pt.u=m||0,v&&b!==u?(this._pt.b=c,this._pt.e=b,this._pt.r=e3):g!==m&&m!=="%"&&(this._pt.b=c,this._pt.r=Jw);else if(p in a)f3.call(this,e,p,c,_?_+u:u);else if(p in e)this.add(e,p,c||e[p],_?_+u:u,s,r);else if(p!=="parseTransform"){Eh(p,u);continue}v||(p in a?E.push(p,0,a[p]):typeof e[p]=="function"?E.push(p,2,e[p]()):E.push(p,1,c||e[p])),o.push(p)}}M&&jx(this)},render:function(e,t){if(t.tween._time||!Nh())for(var n=t._pt;n;)n.r(e,n.d),n=n._next;else t.styles.revert()},get:Ts,aliases:ns,getSetter:function(e,t,n){var s=ns[t];return s&&s.indexOf(",")<0&&(t=s),t in Os&&t!==ri&&(e._gsap.x||Ts(e,"x"))?n&&o0===n?t==="scale"?r3:s3:(o0=n||{})&&(t==="scale"?o3:a3):e.style&&!Mh(e.style[t])?n3:~t.indexOf("-")?i3:Bh(e,t)},core:{_removeProperty:fr,_getMatrix:kh}};oi.utils.checkPrefix=ea;oi.core.getStyleSaver=t_;(function(i,e,t,n){var s=ii(i+","+e+","+t,function(r){Os[r]=1});ii(e,function(r){_i.units[r]="deg",r_[r]=1}),ns[s[13]]=i+","+e,ii(n,function(r){var o=r.split(":");ns[o[1]]=s[o[0]]})})("x,y,z,scale,scaleX,scaleY,xPercent,yPercent","rotation,rotationX,rotationY,skewX,skewY","transform,transformOrigin,svgOrigin,force3D,smoothOrigin,transformPerspective","0:translateX,1:translateY,2:translateZ,8:rotate,8:rotationZ,8:rotateZ,9:rotateX,10:rotateY");ii("x,y,z,top,right,bottom,left,width,height,fontSize,padding,margin,perspective",function(i){_i.units[i]="px"});oi.registerPlugin(l_);var Fr=oi.registerPlugin(l_)||oi;Fr.core.Tween;const _3=(i,e)=>{const t=i.__vccOpts||i;for(const[n,s]of e)t[n]=s;return t},v3={class:"top-hud"},A3={class:"top-actions"},S3={class:"cinematic-head"},y3={class:"cinematic-head-actions"},b3={class:"cinematic-loop-toggle"},M3={class:"cinematic-actions"},C3=["disabled"],T3={class:"cinematic-progress-row"},E3=["value"],w3={class:"cinematic-progress-row"},R3={class:"cinematic-progress-row"},I3={class:"cinematic-focus-toggle"},D3={key:2,class:"fps-counter"},P3={key:0,class:"loading-overlay"},F3={key:1,class:"error-overlay"},L3={class:"error-card"},B3={class:"error-msg"},U3=["min","max"],O3={class:"focal-row"},N3=["min","max"],z3={class:"focal-row"},k3={class:"focal-row"},H3={class:"camera-track-header"},V3={class:"camera-track-copy"},G3=["onClick"],W3=["src"],X3={key:1,class:"camera-tag-overlay"},q3={class:"camera-tag-text"},Y3={key:2},Q3=["src"],K3={key:0,class:"ref-info"},j3={class:"info-tag info-tag--accent"},$3={key:1,class:"ref-info"},Z3={class:"info-tag"},J3={class:"info-tag"},e2={class:"info-tag"},wr=380,m0=.065,g0=.0022,Jl=.08,t2=1,x0=.0055,_0=.0042,n2=1,i2=.35,s2=1.2,r2=8,o2=.26,a2=.1,df=18,_a=6,l2=.45,c2={__name:"GaussianViewer",setup(i){const e=Ft(null),t=Ft(!1),n=Ft(!1),s=Ft(!1),r={FREE:"free",ORBIT:"orbit"},o=Ft(r.FREE),a=Ft([]),l=Ft(""),c=Ft(""),u=Ft(""),f=Ft(""),d=Ft({}),h=Ft({x:0,y:0,z:0}),x=Ft({x:0,y:0,z:0}),p=Ft(""),g=Ft(0),m=Ft(!1),_=Ft(0),v=Ft(0),A=Ft(null),S=Ft(1),y=Ft(0),M=Ft(!0),E=Ft(!1),b=Ft(!1),C=Ft(.68),P=Ft(!0),B=Ft(!1),N=Ir(()=>o.value===r.ORBIT),k=Ir(()=>{if(!l.value.trim()){const L=a.value.filter(re=>re.tag);return L.length>0?L:a.value.slice(0,60)}const I=l.value.trim().toLowerCase();return a.value.filter(L=>L.tag&&L.tag.toLowerCase().includes(I))}),V=()=>{k.value.length>0?D(k.value[0]):alert("场景中没有找到符合该描述的视角哦~")};let F,q;const G=new U(0,1,0);let j=null,ce=!1,ye=!1,Ie=!1,Ne=0;const de={trajectory:null,phase:"main",startTimeMs:0,elapsedMs:0,lastNearestPoseIndex:-1,filteredSample:null},qe=Ft({x:0,y:0}),ee=Ir(()=>a.value.length>=2),ue=Ir(()=>E.value?"暂停运镜":b.value?"继续运镜":"开始运镜"),Me=(I,L)=>!I||!L?null:2*Math.atan(L/2/I)*(180/Math.PI),Ge=(I,L)=>{if(!I||!L)return null;const re=I*Math.PI/180/2;return re<=0?null:L/2/Math.tan(re)},Ce=()=>{if(!F||!F.camera)return;const I=d.value.h||e.value?.clientHeight||window.innerHeight;if(_.value=Number(F.camera.fov||0),I&&_.value>0&&_.value<179){const L=Ge(_.value,I);v.value=L?Number(L.toFixed(1)):0}},Ye=(I,L={})=>{if(!F||!F.camera)return;const re=d.value.h||e.value?.clientHeight||window.innerHeight;if(!re||!I)return;const Se=Me(I,re);if(!Se||!Number.isFinite(Se))return;const Te=F.camera,ze=L.duration??0;if(ze>0)Fr.to(Te,{fov:Se,duration:ze,ease:L.ease||"power2.out",onUpdate:()=>{Te.updateProjectionMatrix();try{F.update(),F.render()}catch{}Ce()}});else{Te.fov=Se,Te.updateProjectionMatrix();try{F.update(),F.render()}catch{}Ce()}},O=I=>Number.isFinite(I)?Math.min(ne.value,Math.max(w.value,I)):null,z=()=>{const I=Number(A.value||v.value||d.value.fl_y||wr);return O(I)},Q=I=>{if(!F||!F.camera||!Number.isFinite(I)||I<=0)return;const L=z();if(!L)return;const re=O(L*I);re&&(A.value=Number(re.toFixed(1)),Ye(re))},w=Ir(()=>{const I=Number(d.value.fl_y||0);return I>0?Math.max(50,Math.floor(I*.4)):50}),ne=Ir(()=>{const I=Number(d.value.fl_y||0);return I>0?Math.max(500,Math.ceil(I*2.5)):3e3}),oe=()=>{m.value=!m.value,m.value&&!A.value&&(A.value=Number((v.value||d.value.fl_y||wr).toFixed(1)))},he=()=>{const I=Number(A.value);!Number.isFinite(I)||I<=0||Ye(I)},ie=()=>{const I=Number(d.value.fl_y||0);I&&(A.value=Number(I.toFixed(1)),Ye(I,{duration:.5,ease:"power2.inOut"}))},pe=()=>{if(!F||!F.camera)return;const I=new Gi().setFromQuaternion(F.camera.quaternion,"YXZ");h.value={x:(I.x*180/Math.PI).toFixed(1),y:(I.y*180/Math.PI).toFixed(1),z:(I.z*180/Math.PI).toFixed(1)},Ce()},te=()=>me.uCenter.value.clone(),ve=()=>{const I=Number(me.uMaxRadius.value||0);return I>0?I:1},R=()=>{Ne&&(cancelAnimationFrame(Ne),Ne=0)},T=()=>{!F||!F.camera||(Fr.killTweensOf(F.camera.position),Fr.killTweensOf(F.camera.quaternion),Fr.killTweensOf(F.camera))},X=I=>{fe(I)},se=I=>I?String(I.id||I.image_id||I.imageId||Nn(I)||JSON.stringify(et(I.matrix)||[])):"",fe=(I,L={})=>{f.value=se(I),L.updateReference!==!1&&(c.value=I?.image_url||Nn(I),u.value=I?.tag||"")},$=(I={})=>{R(),de.trajectory=null,de.startTimeMs=0,de.elapsedMs=0,de.lastNearestPoseIndex=-1,de.filteredSample=null,E.value=!1,b.value=!1,I.resetProgress!==!1&&(y.value=0)},Be=()=>{!E.value&&!b.value||$({resetProgress:!1})},be=(I,L)=>{if(!Array.isArray(I)||I.length<3)return I.map(Qe=>Qe.clone());const re=An.clamp(Number(L)||0,0,1),Se=Math.max(1,Math.round(1+re*3)),Te=.12+re*.26;let ze=I.map(Qe=>Qe.clone());for(let Qe=0;Qe<Se;Qe+=1)ze=ze.map((He,Pe)=>{if(Pe===0||Pe===ze.length-1)return He.clone();const Ze=ze[Pe-1].clone().add(ze[Pe].clone().multiplyScalar(2)).add(ze[Pe+1]).multiplyScalar(.25);return He.clone().lerp(Ze,Te)});return ze},Xe=(I,L)=>{if(!Array.isArray(I)||I.length<3)return I.slice();const re=An.clamp(Number(L)||0,0,1),Se=Math.max(1,Math.round(1+re*2)),Te=.1+re*.28;let ze=I.slice();for(let Qe=0;Qe<Se;Qe+=1)ze=ze.map((He,Pe)=>{if(Pe===0||Pe===ze.length-1)return He;const Ze=(ze[Pe-1]+ze[Pe]*2+ze[Pe+1])/4;return An.lerp(He,Ze,Te)});return ze},H=(I,L)=>{if(L.clone().sub(I).lengthSq()<1e-8)return new Gt;const Se=new rt().lookAt(I,L,G);return new Gt().setFromRotationMatrix(Se)},J=(I,L,re)=>{const Se=Re(L);let Te=I.clone().add(Se);return re&&Se.lengthSq()<1e-8&&(Te=re.clone()),H(I,Te)},_e=I=>{if(!Array.isArray(I)||I.length===0)return[];const L=[I[0].clone().normalize()];for(let re=1;re<I.length;re+=1){const Se=I[re].clone().normalize();L[re-1].dot(Se)<0&&(Se.x*=-1,Se.y*=-1,Se.z*=-1,Se.w*=-1),L.push(Se)}return L},we=(I,L)=>{if(!Array.isArray(I)||I.length<3)return _e(I||[]);const re=An.clamp(Number(L)||0,0,1),Se=Math.max(1,Math.round(1+re*2)),Te=.16+re*.22;let ze=_e(I);for(let Qe=0;Qe<Se;Qe+=1){const nt=ze.map((He,Pe)=>{if(Pe===0||Pe===ze.length-1)return He.clone();const Ze=ze[Pe-1].clone(),ct=ze[Pe].clone(),Lt=ze[Pe+1].clone(),ft=Ze.slerp(Lt,.5);return ct.slerp(ft,Te).normalize()});ze=_e(nt)}return ze},Le=I=>{if(!I)return 0;const L=new U(0,1,0).applyQuaternion(I).normalize();return Math.abs(L.dot(G))},Re=I=>new U(0,0,-1).applyQuaternion(I).normalize(),$e=I=>{if(!Array.isArray(I)||I.length<=df)return I;const L=I.filter(He=>Le(He.quaternion)>=l2),re=L.length>=_a?L:I.slice();if(re.length<=df)return re;const Se=re.map((He,Pe,Ze)=>{const ct=Ze[Math.max(0,Pe-1)],Lt=Ze[Math.min(Ze.length-1,Pe+1)],ft=Le(He.quaternion),sn=Pe>0?He.position.distanceTo(ct.position):0,At=Pe<Ze.length-1?He.position.distanceTo(Lt.position):0,$n=(sn+At)*.5,zn=Re(ct.quaternion),Bi=Re(He.quaternion),fs=Re(Lt.quaternion),xl=Pe>0&&Pe<Ze.length-1?Math.max(0,zn.dot(Bi))*.5+Math.max(0,Bi.dot(fs))*.5:1;return{frame:He,index:Pe,score:ft*2.2+xl*1.4+Math.min($n,1.5)*.4}}),Te=new Set([0,re.length-1]),ze=Math.max(_a,Math.min(df,re.length)),Qe=Se.filter(({index:He})=>Te.has(He)).map(({frame:He})=>He),nt=Se.filter(({index:He})=>!Te.has(He)).sort((He,Pe)=>Pe.score-He.score);for(const He of nt){if(Qe.length>=ze)break;Qe.push(He.frame)}if(Qe.sort((He,Pe)=>He.index-Pe.index),Qe.length<_a){const He=Math.max(1,Math.floor(re.length/_a));for(let Pe=0;Pe<re.length&&Qe.length<_a;Pe+=He){const Ze=re[Pe];Qe.includes(Ze)||Qe.push(Ze)}Qe.sort((Pe,Ze)=>Pe.index-Ze.index)}return Qe},W=({keyframes:I,positions:L,targets:re,focals:Se,durationMs:Te})=>{const ze=ve(),Qe=we(L.map((ct,Lt)=>J(ct,I[Lt].quaternion,re[Lt])),C.value),nt=I.map((ct,Lt)=>({...ct,position:L[Lt],target:re[Lt],stabilizedQuaternion:Qe[Lt],fl_y:Se[Lt]||ct.fl_y})),He=new $p(nt.map(ct=>ct.position.clone()),!1,"centripetal"),Pe=new $p(nt.map(ct=>ct.target.clone()),!1,"centripetal"),Ze=[0];for(let ct=1;ct<nt.length;ct+=1){const Lt=nt[ct-1],ft=nt[ct];Ze.push(Ze[ct-1]+Lt.position.distanceTo(ft.position))}return{keyframes:nt,curve:He,lookCurve:Pe,cumulativeDistances:Ze,totalDistance:Math.max(Ze[Ze.length-1],1e-5),durationMs:Te,lookAheadDistance:An.clamp(ze*(.4+C.value*.45),s2,r2)}},Fe=(I,L)=>{if(!I?.keyframes||I.keyframes.length<2)return null;const re=ve(),Se=I.keyframes[0],Te=I.keyframes[I.keyframes.length-1],ze=Te.position.distanceTo(Se.position);if(ze<1e-4)return null;const Qe=Math.max(re*.55,ze*.22,.9),nt=Math.max(re*.18,ze*.08,.35),He=Te.position.clone().sub(L).setY(0),Pe=Se.position.clone().sub(L).setY(0);He.lengthSq()<1e-6&&He.set(1,0,0),Pe.lengthSq()<1e-6&&Pe.set(-1,0,0),He.normalize().multiplyScalar(nt),Pe.normalize().multiplyScalar(nt);const Ze=L.clone().add(new U(0,re*.15,0)),ct=[Te.position.clone(),Te.position.clone().add(new U(0,Qe,0)).add(He),Se.position.clone().add(new U(0,Qe*.86,0)).add(Pe),Se.position.clone()],Lt=[Te.target.clone().lerp(Ze,.4),Ze.clone(),Ze.clone(),Se.target.clone().lerp(Ze,.28)],ft=Math.max(0,Number(Te.fl_y||Se.fl_y||d.value.fl_y||wr)),sn=An.clamp(ze*1350+1800,2400,6200)/S.value;return W({keyframes:[{index:Te.index,pose:Te.pose,fl_y:ft,h:Te.h},{index:Te.index,pose:Te.pose,fl_y:ft,h:Te.h},{index:Se.index,pose:Se.pose,fl_y:ft,h:Se.h},{index:Se.index,pose:Se.pose,fl_y:ft,h:Se.h}],positions:ct,targets:Lt,focals:[ft,ft,ft,ft],durationMs:sn})},Ae={FLY_IN:0,DIFFUSION:1,COLORING:2,FINISHED:3},xe={isLoaded:!1,lastFrameTime:0,phase:Ae.FLY_IN,flyDuration:1.5,diffusionDuration:1,colorDuration:4},me={uTime:{value:0},uCenter:{value:new U(0,0,0)},uGeoRadius:{value:0},uColorRadius:{value:0},uMaxRadius:{value:50},uParticleProgress:{value:0}},ge=I=>{if(!F)return;const L=I.getSplatCount();I.updateMatrixWorld();let re=1/0,Se=1/0,Te=1/0,ze=-1/0,Qe=-1/0,nt=-1/0;const He=new U,Pe=Math.max(1,Math.floor(L/1e3));for(let gr=0;gr<L;gr+=Pe)I.getSplatCenter(gr,He),He.applyMatrix4(I.matrixWorld),He.x<re&&(re=He.x),He.x>ze&&(ze=He.x),He.y<Se&&(Se=He.y),He.y>Qe&&(Qe=He.y),He.z<Te&&(Te=He.z),He.z>nt&&(nt=He.z);const Ze=(re+ze)/2,ct=(Se+Qe)/2,Lt=(Te+nt)/2,ft=Math.max(ze-re,Qe-Se,nt-Te);me.uCenter.value.set(Ze,ct,Lt),me.uMaxRadius.value=ft*.7;let sn=6e4;L<4e4?sn=L:L>1e6&&(sn=4e5);const At=Math.ceil(L/sn);let $n=ft/200*window.devicePixelRatio;$n<.5&&($n=.5);const zn=ft*1;console.log(`[Adaptive] MaxDim: ${ft.toFixed(2)}, Particles: ~${Math.floor(L/At)}, Size: ${$n.toFixed(2)}`);const Bi=new jn,fs=[],xl=[],Hh=[];for(let gr=0;gr<L;gr+=At){I.getSplatCenter(gr,He),He.applyMatrix4(I.matrixWorld),xl.push(He.x,He.y,He.z);const iu=zn+Math.random()*(ft*.5),Vh=Math.random()*Math.PI*2,su=Math.acos(2*Math.random()-1),u_=Ze+iu*Math.sin(su)*Math.cos(Vh),f_=ct+iu*Math.sin(su)*Math.sin(Vh),d_=Lt+iu*Math.cos(su);fs.push(u_,f_,d_),Hh.push(Math.random())}Bi.setAttribute("position",new Bn(fs,3)),Bi.setAttribute("aTarget",new Bn(xl,3)),Bi.setAttribute("aRandom",new Bn(Hh,1));const c_=new Kn({uniforms:{uProgress:me.uParticleProgress,uSize:{value:$n},uColor:{value:new vt(.6,.6,.6)}},vertexShader:`
      uniform float uProgress;
      uniform float uSize;
      attribute vec3 aTarget;
      attribute float aRandom;
      
      float easeOutCubic(float x) { return 1.0 - pow(1.0 - x, 3.0); }
      
      void main() {
        float t = (uProgress - aRandom * 0.1) / 0.9;
        t = clamp(t, 0.0, 1.0);
        vec3 pos = mix(position, aTarget, easeOutCubic(t));
        
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        
        // 距离衰减 (20.0 是透视缩放因子，配合世界单位的 uSize 使用)
        gl_PointSize = uSize * (20.0 / -mvPosition.z);
        if(gl_PointSize < 1.0) gl_PointSize = 1.0;
      }
    `,fragmentShader:`
      uniform vec3 uColor;
      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if(length(coord) > 0.5) discard;
        gl_FragColor = vec4(uColor, 1.0);
      }
    `,transparent:!0,opacity:1,depthTest:!0,depthWrite:!1});q=new Ty(Bi,c_),q.frustumCulled=!1,F.threeScene.add(q)},We=I=>{if(!I||!I.material)return;const L=I.material;L.uniforms=L.uniforms||{},L.uniforms.uGeoRadius=me.uGeoRadius,L.uniforms.uColorRadius=me.uColorRadius,L.uniforms.uMaxRadius=me.uMaxRadius,L.uniforms.uCenter=me.uCenter,L.vertexShader=`varying vec3 vWorldPosition;
`+L.vertexShader;const re=L.vertexShader.lastIndexOf("}");if(re!==-1){const ze=`vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
`;L.vertexShader=L.vertexShader.substring(0,re)+ze+"}"}const Se=`
    uniform float uGeoRadius;
    uniform float uColorRadius;
    uniform float uMaxRadius;
    uniform vec3 uCenter;
    varying vec3 vWorldPosition;
  `;L.fragmentShader=Se+L.fragmentShader;const Te=L.fragmentShader.lastIndexOf("}");if(Te!==-1){const ze=L.fragmentShader.substring(0,Te),Qe=`
      float distFromCenter = distance(vWorldPosition, uCenter);
      
      if (distFromCenter > uGeoRadius) {
          discard;
      }
      if (distFromCenter > uColorRadius) {
          if (gl_FragColor.a < 0.8) discard; 
          gl_FragColor.a = 1.0; 
          gl_FragColor.rgb = vec3(0.6, 0.6, 0.6);
      } 
    `;L.fragmentShader=ze+Qe+"}"}L.needsUpdate=!0},et=I=>{if(!Array.isArray(I))return null;if(I.length===16){const L=I.map(re=>Number(re));return L.every(Number.isFinite)?L:null}if(I.length===4&&I.every(L=>Array.isArray(L)&&L.length===4)){const L=I.flat().map(re=>Number(re));return L.every(Number.isFinite)?L:null}return null},Dt=I=>{if(I==null)return"";let L=String(I).trim();if(!L)return"";try{L=decodeURIComponent(L)}catch{}L=L.replace(/\\/g,"/");const re=L.split("/");return(re[re.length-1]||"").trim().toLowerCase()},Ct=I=>/^https?:\/\//i.test(String(I||"")),Si=I=>{if(typeof I!="string"||!I.trim())return"";const L=I.trim();return Ct(L)&&window.location.origin.startsWith("http://127.0.0.1:")?`${window.location.origin}/proxy/${encodeURIComponent(L)}`:L},Nn=I=>{if(!I)return"";const L=I.id||I.image_id||I.imageId;if(L)return Dt(L);const re=I.image_url;if(typeof re!="string"||re.length===0)return"";const Se=re.split("?")[0];return Dt(Se)},Zc=I=>{if(!I||a.value.length===0)return null;const L=Dt(I.imageId);if(L){const ze=a.value.find(Qe=>Nn(Qe)===L);if(ze)return ze}const re=et(I.matrix);if(!re)return null;let Se=null,Te=Number.POSITIVE_INFINITY;for(const ze of a.value){const Qe=et(ze.matrix);if(!Qe)continue;let nt=0;for(let He=0;He<16;He+=1){const Pe=Math.abs(Qe[He]-re[He]);if(Pe>nt&&(nt=Pe),nt>=Te)break}nt<Te&&(Te=nt,Se=ze)}return Te<=1e-4?Se:null},us=(I=!1)=>{if(!j||ce)return;if(!j.imageId){const Se=Wi();if(Se){ce=!0,D(Se);return}}const L=Zc(j);if(L){ce=!0,D(L);return}if(!I||j.imageId&&!Ie)return;const re=et(j.matrix);re&&(ce=!0,D({matrix:re,image_url:j.imageId||""}))},sa=I=>{const L=I?.image_url;return typeof L=="string"&&L.trim().length>0},Wi=()=>{if(!Array.isArray(a.value)||a.value.length===0)return null;const I=a.value.find(re=>sa(re)&&re.tag);if(I)return I;const L=a.value.find(re=>sa(re));return L||a.value[0]||null},ra=()=>{if(!Array.isArray(k.value)||k.value.length===0)return a.value;const I=k.value.filter(re=>typeof re?.tag=="string"&&re.tag.trim().length>0);if(I.length>=2)return I.slice(0,12);if(k.value.length>=2)return k.value.slice(0,12);const L=a.value.filter(re=>typeof re?.tag=="string"&&re.tag.trim().length>0);return L.length>=2?L.slice(0,12):a.value.slice(0,12)},$r=()=>{if(j||ce||ye)return;const I=Wi();I&&(ye=!0,D(I))},oa=I=>{if(!F||!F.camera)return null;const L=et(I?.matrix);if(!L)return null;const re=F.getSplatMesh(),Se=new rt().fromArray(L),Te=new rt;re?(re.updateMatrixWorld(),Te.copy(re.matrixWorld).multiply(Se)):Te.copy(Se);const ze=new U,Qe=new Gt,nt=new U;return Te.decompose(ze,Qe,nt),{position:ze,quaternion:Qe,fl_y:Number(I?.fl_y||d.value.fl_y||0),h:Number(I?.h||d.value.h||0)}},mr=()=>{const I=ra();if(!F||!Array.isArray(I)||I.length<2)return null;const L=I.map((At,$n)=>{const zn=oa(At);return zn?{index:$n,pose:At,position:zn.position,quaternion:zn.quaternion,fl_y:zn.fl_y,h:zn.h}:null}).filter(Boolean);if(L.length<2)return null;const re=[L[0]];for(let At=1;At<L.length;At+=1){const $n=re[re.length-1],zn=L[At],Bi=$n.position.distanceToSquared(zn.position)<1e-6,fs=Math.abs($n.quaternion.dot(zn.quaternion))>.999999;Bi&&fs||re.push(zn)}if(re.length<2)return null;const Se=$e(re);if(Se.length<2)return null;const Te=te(),ze=Se,Qe=ze.map(At=>At.position.clone()),nt=be(Qe,C.value),He=Xe(ze.map(At=>At.fl_y||0),C.value),Pe=ze.map((At,$n)=>{const zn=new U(0,0,-1).applyQuaternion(At.quaternion).normalize(),Bi=Math.max(.8,Qe[$n].distanceTo(Te)),fs=Qe[$n].clone().add(zn.multiplyScalar(Math.max(2.2,Bi*.9)));return P.value?fs.lerp(Te,An.clamp(.48+C.value*.26,0,.9)):fs}),Ze=be(Pe,C.value);let ct=0;for(let At=1;At<nt.length;At+=1)ct+=nt[At-1].distanceTo(nt[At]);const Lt=ze.length-1,ft=An.clamp(ct*1600+Lt*260,7e3,42e3)/S.value,sn=W({keyframes:ze,positions:nt,targets:Ze,focals:He,durationMs:ft});return{...sn,worldCenter:Te.clone(),loopBridge:Fe(sn,Te)}},Zr=(I,L)=>{if(!I)return null;const re=An.clamp(L,0,1),Se=I.totalDistance*re;let Te=I.keyframes.length-2;for(let At=0;At<I.cumulativeDistances.length-1;At+=1)if(Se<=I.cumulativeDistances[At+1]){Te=At;break}const ze=I.cumulativeDistances[Te],Qe=I.cumulativeDistances[Te+1],nt=Math.max(Qe-ze,1e-5),He=An.smootherstep((Se-ze)/nt,0,1),Pe=I.keyframes[Te],Ze=I.keyframes[Te+1],ct=I.curve.getPointAt(re),Lt=Pe.stabilizedQuaternion.clone().slerp(Ze.stabilizedQuaternion,He).normalize(),ft=Pe.target.clone().lerp(Ze.target,He);return{position:ct,quaternion:Lt,target:ft,fl_y:Pe.fl_y&&Ze.fl_y?An.lerp(Pe.fl_y,Ze.fl_y,He):Pe.fl_y||Ze.fl_y||0,h:Pe.h||Ze.h||d.value.h||0,nearestPoseIndex:He<.5?Pe.index:Ze.index}},zs=I=>{if(!I||!F||!F.camera)return;const L=An.lerp(o2,a2,C.value);de.filteredSample?(de.filteredSample.position.lerp(I.position,L),de.filteredSample.quaternion.slerp(I.quaternion,L).normalize(),I.fl_y&&(de.filteredSample.fl_y=An.lerp(de.filteredSample.fl_y||I.fl_y,I.fl_y,L*.85)),I.h&&(de.filteredSample.h=I.h)):de.filteredSample={position:I.position.clone(),quaternion:I.quaternion.clone(),fl_y:Number(I.fl_y||0),h:Number(I.h||d.value.h||0)};const re=F.camera;if(re.position.copy(de.filteredSample.position),re.quaternion.copy(de.filteredSample.quaternion),de.filteredSample.fl_y&&de.filteredSample.h?(d.value.h=de.filteredSample.h,A.value=Number(de.filteredSample.fl_y.toFixed(1)),Ye(de.filteredSample.fl_y)):Ve(),I.nearestPoseIndex!==de.lastNearestPoseIndex){de.lastNearestPoseIndex=I.nearestPoseIndex;const Se=a.value[I.nearestPoseIndex];Se&&fe(Se,{updateReference:!1})}},aa=I=>{if(!de.trajectory||!F||!F.camera){$({resetProgress:!1});return}const L=de.phase==="loop-bridge"&&de.trajectory.loopBridge?de.trajectory.loopBridge:de.trajectory,re=Math.max(L.durationMs,1),Se=Math.max(0,I-de.startTimeMs);de.elapsedMs=Se;let Te=Se/re;if(Te>=1&&(de.phase==="loop-bridge"?(de.startTimeMs=I,de.elapsedMs=0,de.phase="main",de.lastNearestPoseIndex=-1,Te=0):M.value&&de.trajectory.loopBridge?(de.startTimeMs=I,de.elapsedMs=0,de.phase="loop-bridge",de.lastNearestPoseIndex=-1,Te=0):M.value?(de.startTimeMs=I,de.elapsedMs=0,de.phase="main",de.lastNearestPoseIndex=-1,Te=0):Te=1),y.value=de.phase==="main"?Te:1,zs(Zr(L,Te)),!M.value&&de.phase==="main"&&Te>=1){$({resetProgress:!1}),y.value=1;return}Ne=requestAnimationFrame(aa)},pl=(I={})=>{if(!F||!F.camera)return;const L=mr();L&&(T(),R(),de.trajectory=L,de.phase="main",de.filteredSample=null,de.elapsedMs=I.resume?de.elapsedMs:0,de.startTimeMs=performance.now()-de.elapsedMs,de.lastNearestPoseIndex=-1,E.value=!0,b.value=!1,I.resume||(y.value=0,zs(Zr(L,0))),Ne=requestAnimationFrame(aa))},Jc=()=>{E.value&&(R(),de.elapsedMs=Math.max(0,performance.now()-de.startTimeMs),E.value=!1,b.value=!0)},eu=()=>{if(ee.value){if(E.value){Jc();return}pl({resume:b.value})}},tu=()=>{ee.value&&(B.value=!B.value)},ml=()=>{const I=mr();I&&(de.trajectory=I,de.phase="main",de.lastNearestPoseIndex=-1,zs(Zr(I,y.value)),E.value?(de.elapsedMs=I.durationMs*y.value,de.startTimeMs=performance.now()-de.elapsedMs):b.value&&(de.elapsedMs=I.durationMs*y.value))},nu=()=>{S.value=Number(An.clamp(Number(S.value)||1,.25,3).toFixed(2)),(E.value||b.value)&&ml()},gl=()=>{C.value=Number(An.clamp(Number(C.value)||.68,0,1).toFixed(2)),(E.value||b.value)&&ml()},D=(I,L={})=>{if(!F||!F.camera)return;const re=oa(I);if(!re){console.warn("[Viewer] Skip invalid pose matrix:",I);return}L.keepCinematic||Be();const Se=F.camera,Te=re.position,ze=re.quaternion;X(I);const Qe=re.fl_y,nt=re.h;Qe&&nt&&(d.value.h=nt,A.value=Number(Qe.toFixed(1)),Ye(Qe,{duration:1.5,ease:"power3.inOut"})),Se.near>.001&&(Se.near=.001,Se.updateProjectionMatrix()),t.value=!1,F.controls&&(F.controls.enabled=!1);const He=Se.position.clone(),Pe=Se.quaternion.clone(),Ze={t:0};T(),Fr.killTweensOf(Ze),Fr.to(Ze,{t:1,duration:1.5,ease:"power3.inOut",onUpdate:()=>{Se.position.lerpVectors(He,Te,Ze.t),Se.quaternion.slerpQuaternions(Pe,ze,Ze.t)},onComplete:()=>{const ct=new Gi().setFromQuaternion(Se.quaternion,"YXZ");x.value={x:(ct.x*180/Math.PI).toFixed(1),y:(ct.y*180/Math.PI).toFixed(1),z:(ct.z*180/Math.PI).toFixed(1)},qe.value={x:0,y:0},Pt.roll=0,pe(),F.controls&&(F.controls.enabled=!0)}})},Y=()=>{const I=/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);return{rootElement:e.value,cameraUp:[0,1,0],initialCameraPosition:[0,0,5],initialCameraLookAt:[0,0,0],useBuiltInControls:!1,gpuAcceleratedSort:!1,webXRMode:Dr.None,sharedMemoryForWorkers:!1,antialiased:!I}};let le="/models/scene_auto_sync.ply",ae="/models/webgl_poses_with_tags.json",K=!1;const Ee=()=>{const I=new URLSearchParams(window.location.search),L=I.get("payload");if(L)try{const nt=JSON.parse(decodeURIComponent(L));return{ply:nt.ply||null,poses:nt.poses||null,matrix:nt.matrix||null,imageId:nt.imageId||null}}catch(nt){console.warn("[Viewer] 无法解析 payload 查询参数:",nt)}const re=I.get("ply"),Se=I.get("poses"),Te=I.get("matrix"),ze=I.get("imageId");let Qe=null;if(Te)try{Qe=JSON.parse(decodeURIComponent(Te))}catch(nt){console.warn("[Viewer] 无法解析 matrix 查询参数:",nt)}return re||Se||Qe?{ply:re||null,poses:Se||null,matrix:Qe,imageId:ze||null}:null},Ue=async(I,L,re)=>{if(!n.value){n.value=!0,$(),I&&(le=I),L&&(ae=L);try{F&&(F.renderer.setAnimationLoop(null),F.dispose&&await F.dispose(),F=null),e.value&&(e.value.innerHTML=""),xe.isLoaded=!1,xe.phase=Ae.FLY_IN,me.uParticleProgress.value=0,me.uGeoRadius.value=0,me.uColorRadius.value=0,j=null,ce=!1,ye=!1,Ie=!1;const Se=Y();F=new bo(Se),window.viewer=F,A.value=wr,console.log(`[Viewer] 加载模型: ${le}`),await F.addSplatScene(le,{showLoadingUI:!0,progressiveLoad:!1,rotation:[0,0,0,1]}),n.value=!1,window.BrainDanceChannel&&window.BrainDanceChannel.postMessage(JSON.stringify({status:"success",msg:"模型加载完成"})),console.log(`[Viewer] 加载位姿: ${ae}`),fetch(ae).then(Pe=>Pe.json()).then(Pe=>{Ie=!0,Pe.frames?(d.value={w:Pe.w,h:Pe.h,fl_x:Pe.fl_x,fl_y:Pe.fl_y},A.value=Number((Pe.fl_y||0).toFixed(1)),a.value=Pe.frames.map(Ze=>{let ct=Ze.image_url;if(ct&&!ct.startsWith("http")&&ae.startsWith("http")){const Lt=ae.substring(0,ae.lastIndexOf("/"));let ft=ct;const sn=ft.indexOf("images/");sn!==-1?ft=ft.substring(sn):ft.startsWith("/models/")?ft=ft.substring(8):ft.startsWith("/")&&(ft=ft.substring(1)),ct=`${Lt}/${ft}`}return ct=Si(ct),{id:Ze.id,matrix:Ze.matrix,image_url:ct,tag:Ze.tag,fl_x:Ze.fl_x,fl_y:Ze.fl_y,w:Ze.w||Pe.w,h:Ze.h||Pe.h}}),d.value.fl_y&&d.value.h?Ye(d.value.fl_y):Ye(wr),us(!0),$r()):(a.value=Pe,Ye(wr),us(!0),$r())}).catch(Pe=>{Ie=!0,console.error("加载位姿失败:",Pe),Ye(wr),us(!0)});const Te=F.getSplatMesh();Te.visible=!1,setTimeout(()=>{Te&&(ge(Te),We(Te),re&&(re.matrix||re.imageId)?(j={matrix:re.matrix||null,imageId:re.imageId||null},us(Ie),setTimeout(()=>{us(!1)},50),re.imageId||setTimeout(()=>{us(!0)},800)):setTimeout(()=>{$r()},80),xe.lastFrameTime=Date.now(),xe.startTime=Date.now(),xe.isLoaded=!0)},200);let ze=performance.now();const Qe=1e3/120;let nt=0,He=performance.now();F.renderer.setAnimationLoop(()=>{const Pe=performance.now(),Ze=Pe-ze;if(Ze<Qe||(ze=Pe-Ze%Qe,F.update(),F.render(),nt++,Pe-He>=1e3&&(g.value=nt,nt=0,He=Pe),!xe.isLoaded||xe.phase===Ae.FINISHED))return;const ct=Date.now(),Lt=(ct-xe.lastFrameTime)/1e3||.016;if(xe.lastFrameTime=ct,xe.phase===Ae.FLY_IN){const ft=1/xe.flyDuration;let sn=me.uParticleProgress.value+Lt*ft;if(sn>=1.2){sn=1.2;const At=F.getSplatMesh();At&&(At.visible=!0),xe.phase=Ae.DIFFUSION,xe.diffuseTime=0}me.uParticleProgress.value=sn}else if(xe.phase===Ae.DIFFUSION){xe.diffuseTime+=Lt;const ft=Math.min(xe.diffuseTime/xe.diffusionDuration,1),sn=me.uMaxRadius.value;me.uGeoRadius.value=ft*(sn*1.5),q&&q.material&&(q.material.opacity=1-ft),ft>=1&&(q&&(q.visible=!1),me.uGeoRadius.value=99999,xe.phase=Ae.COLORING,xe.colorStartTime=ct)}else if(xe.phase===Ae.COLORING){const ft=(ct-xe.colorStartTime)/1e3,sn=me.uMaxRadius.value,At=ft/xe.colorDuration;me.uColorRadius.value=At*(sn*1.5),At>=1&&(xe.phase=Ae.FINISHED,me.uColorRadius.value=99999)}}),Et()}catch(Se){console.error("error:",Se),p.value=Se&&(Se.message||String(Se))||"模型加载失败，请检查模型 URL 是否正确可访问"}finally{n.value=!1}}},je=()=>{!F||!F.controls||(F.controls.dispose(),F.controls=null)},Ve=()=>{if(!(!F||!F.camera)){F.camera.updateProjectionMatrix(),Ce(),pe();try{F.update(),F.render()}catch{}}},st=(I,L)=>{!F||!F.camera||(F.camera.rotateOnWorldAxis(G,-I),F.camera.rotateX(-L),Ve())},ot=I=>{!F||!F.camera||!Number.isFinite(I)||(F.camera.rotateZ(I*n2),Ve())},Je=I=>{if(!F||!F.camera||!Number.isFinite(I)||I<=0)return;const L=Math.max(.3,F.camera.position.distanceTo(te())),re=An.clamp((1-I)*L*i2,-L*.25,L*.25);F.camera.translateZ(re),Ve()},mt=(I,L)=>Math.atan2(L.clientY-I.clientY,L.clientX-I.clientX),Tt=I=>I>Math.PI?I-Math.PI*2:I<-Math.PI?I+Math.PI*2:I,Yt=()=>{F&&je()},Qt=()=>{F&&(je(),Pt.roll=0)},Et=()=>{F&&(N.value?Qt():Yt())},tt=I=>{I!==r.FREE&&I!==r.ORBIT||o.value!==I&&(o.value=I,Et(),N.value)},Xt=()=>{const I=window.location.hostname==="localhost"||window.location.hostname==="127.0.0.1",L=window.location.protocol==="https:";s.value=I||L},it=Ft(!1),dt={x:0,y:0},xt={active:!1,distance:0},Pt={active:!1,angle:0,roll:0},yi=(I,L)=>{const re=I.clientX-L.clientX,Se=I.clientY-L.clientY;return Math.hypot(re,Se)},zt=I=>{if(Be(),N.value){if(I.button!==0)return;it.value=!0,xt.active=!1,Pt.active=!1,dt.x=I.clientX,dt.y=I.clientY;return}it.value=!0,xt.active=!1,dt.x=I.clientX,dt.y=I.clientY},xn=I=>{if(N.value){if(!it.value||!F||!F.camera)return;const Te=I.clientX-dt.x,ze=I.clientY-dt.y;st(Te*x0,ze*_0),dt.x=I.clientX,dt.y=I.clientY;return}if(!it.value||!F||!F.camera)return;const L=I.clientX-dt.x,Se=(I.clientY-dt.y)*m0;F.camera.rotateX(Se*Math.PI/180),F.camera.translateX(-L*g0),F.camera.updateProjectionMatrix(),pe(),dt.x=I.clientX,dt.y=I.clientY},yn=()=>{if(N.value){it.value=!1,xt.active=!1,Pt.active=!1;return}it.value=!1,xt.active=!1},bn=I=>{if(!F||!F.camera)return;if(Be(),N.value){const re=I.deltaY<0?1+Jl:1/(1+Jl);Je(re);return}const L=I.deltaY<0?1+Jl:1/(1+Jl);Q(L)},Rn=I=>{if(Be(),N.value){if(I.touches.length>=2){it.value=!1,xt.active=!0,xt.distance=yi(I.touches[0],I.touches[1]),Pt.active=!0,Pt.angle=mt(I.touches[0],I.touches[1]);return}xt.active=!1,Pt.active=!1,I.touches.length===1&&(it.value=!0,dt.x=I.touches[0].clientX,dt.y=I.touches[0].clientY);return}if(I.touches.length>=2){it.value=!1,xt.active=!0,xt.distance=yi(I.touches[0],I.touches[1]);return}xt.active=!1,I.touches.length===1&&(it.value=!0,dt.x=I.touches[0].clientX,dt.y=I.touches[0].clientY)},la=I=>{if(N.value){if(!F||!F.camera||I.touches.length===0)return;if(I.touches.length>=2){const Qe=yi(I.touches[0],I.touches[1]),nt=mt(I.touches[0],I.touches[1]);xt.active&&xt.distance>0&&Qe>0&&Je(Qe/xt.distance),Pt.active&&ot(Tt(nt-Pt.angle)),xt.active=!0,xt.distance=Qe,Pt.active=!0,Pt.angle=nt,it.value=!1;return}if(!it.value)return;const Te=I.touches[0].clientX-dt.x,ze=I.touches[0].clientY-dt.y;st(Te*x0,ze*_0),dt.x=I.touches[0].clientX,dt.y=I.touches[0].clientY;return}if(!F||!F.camera||I.touches.length===0)return;if(I.touches.length>=2){const Te=yi(I.touches[0],I.touches[1]);if(xt.active&&xt.distance>0&&Te>0){const ze=Te/xt.distance;Q(1+(ze-1)*t2)}xt.active=!0,xt.distance=Te,it.value=!1;return}if(!it.value)return;const L=I.touches[0].clientX-dt.x,Se=(I.touches[0].clientY-dt.y)*m0;qe.value.x+=Se,F.camera.rotateX(Se*Math.PI/180),F.camera.translateX(-L*g0),F.camera.updateProjectionMatrix(),pe(),dt.x=I.touches[0].clientX,dt.y=I.touches[0].clientY},Xi=I=>{if(N.value){if(I.touches.length>=2){xt.active=!0,xt.distance=yi(I.touches[0],I.touches[1]),Pt.active=!0,Pt.angle=mt(I.touches[0],I.touches[1]),it.value=!1;return}xt.active=!1,xt.distance=0,Pt.active=!1,Pt.angle=0,it.value=!1,I.touches.length===1&&(dt.x=I.touches[0].clientX,dt.y=I.touches[0].clientY,it.value=!0);return}if(I.touches.length>=2){xt.active=!0,xt.distance=yi(I.touches[0],I.touches[1]),it.value=!1;return}xt.active=!1,xt.distance=0,it.value=!1,I.touches.length===1&&(dt.x=I.touches[0].clientX,dt.y=I.touches[0].clientY,it.value=!0)};return eg(()=>{if(e.value){if(Xt(),window.loadModelFromFlutter=I=>{console.log("[Flutter->WebGL] 收到加载请求:",I),typeof I=="string"?Ue(I,null,null):typeof I=="object"&&I!==null?Ue(I.ply||null,I.poses||null,{matrix:I.matrix||null,imageId:I.imageId||null}):Ue(null,null,null)},window.BrainDanceChannel)window.BrainDanceChannel.postMessage(JSON.stringify({status:"ready"}));else{const I=Ee();I&&!K?(K=!0,Ue(I.ply,I.poses,{matrix:I.matrix||null,imageId:I.imageId||null})):Ue(null,null)}window.addEventListener("mousedown",zt),window.addEventListener("mousemove",xn),window.addEventListener("mouseup",yn)}}),tg(async()=>{window.removeEventListener("mousedown",zt),window.removeEventListener("mousemove",xn),window.removeEventListener("mouseup",yn),$(),F&&(F.renderer.setAnimationLoop(null),await F.dispose())}),(I,L)=>(vn(),Cn("div",{class:"app-container",onMousedown:zt,onMousemove:xn,onMouseup:yn,onWheel:bt(bn,["prevent"]),onMouseleave:yn,onTouchstart:Rn,onTouchmove:bt(la,["prevent"]),onTouchend:Xi,onTouchcancel:Xi},[Oe("div",{ref_key:"containerRef",ref:e,class:"viewer-container"},null,512),L[58]||(L[58]=Oe("div",{class:"viewer-vignette"},null,-1)),Oe("div",v3,[Oe("div",{class:"search-panel archive-card",onMousedown:L[1]||(L[1]=bt(()=>{},["stop"])),onTouchstart:L[2]||(L[2]=bt(()=>{},["stop"])),onTouchmove:L[3]||(L[3]=bt(()=>{},["stop"])),onTouchend:L[4]||(L[4]=bt(()=>{},["stop"]))},[xr(Oe("input",{type:"text","onUpdate:modelValue":L[0]||(L[0]=re=>l.value=re),onKeyup:BA(V,["enter"]),placeholder:"例如：门口、桌面左侧、正面特写",class:"search-input"},null,544),[[fa,l.value]]),Oe("button",{onClick:V,class:"archive-btn archive-btn--solid search-btn"},"检索视角")],32),Oe("div",A3,[Oe("div",{class:"view-mode-switch archive-card",onMousedown:L[7]||(L[7]=bt(()=>{},["stop"])),onTouchstart:L[8]||(L[8]=bt(()=>{},["stop"])),onTouchmove:L[9]||(L[9]=bt(()=>{},["stop"])),onTouchend:L[10]||(L[10]=bt(()=>{},["stop"]))},[Oe("button",{class:Or(["mode-chip",{active:o.value===r.FREE}]),onClick:L[5]||(L[5]=re=>tt(r.FREE))}," 自由模式 ",2),Oe("button",{class:Or(["mode-chip",{active:o.value===r.ORBIT}]),onClick:L[6]||(L[6]=re=>tt(r.ORBIT))}," Orbit 模式 ",2)],32),Oe("button",{class:"archive-btn archive-btn--ghost focal-settings-toggle",onClick:oe,onMousedown:L[11]||(L[11]=bt(()=>{},["stop"])),onTouchstart:L[12]||(L[12]=bt(()=>{},["stop"])),onTouchend:L[13]||(L[13]=bt(()=>{},["stop"]))},Mn(m.value?"收起焦距":"焦距设置"),33),ee.value?(vn(),Cn("button",{key:0,class:Or(["cinematic-trigger archive-btn archive-btn--ghost",{active:B.value}]),onClick:tu,onMousedown:L[14]||(L[14]=bt(()=>{},["stop"])),onTouchstart:L[15]||(L[15]=bt(()=>{},["stop"])),onTouchend:L[16]||(L[16]=bt(()=>{},["stop"]))},[...L[41]||(L[41]=[Oe("span",{class:"cinematic-trigger-icon","aria-hidden":"true"},[Oe("svg",{viewBox:"0 0 24 24",focusable:"false"},[Oe("path",{d:"M4 7.5a1.5 1.5 0 0 1 1.5-1.5h7A1.5 1.5 0 0 1 14 7.5v9a1.5 1.5 0 0 1-1.5 1.5h-7A1.5 1.5 0 0 1 4 16.5v-9Zm11 2.1 4.83-2.76A.75.75 0 0 1 21 7.5v9a.75.75 0 0 1-1.17.66L15 14.4V9.6Z"})])],-1),Oe("span",null,"运镜",-1)])],34)):ai("",!0),ee.value&&B.value?(vn(),Cn("div",{key:1,class:"cinematic-panel archive-card",onMousedown:L[23]||(L[23]=bt(()=>{},["stop"])),onTouchstart:L[24]||(L[24]=bt(()=>{},["stop"])),onTouchmove:L[25]||(L[25]=bt(()=>{},["stop"])),onTouchend:L[26]||(L[26]=bt(()=>{},["stop"])),onTouchcancel:L[27]||(L[27]=bt(()=>{},["stop"]))},[Oe("div",S3,[L[43]||(L[43]=Oe("div",null,[Oe("div",{class:"eyebrow"},"Camera Move"),Oe("div",{class:"cinematic-title"},"自动运镜")],-1)),Oe("div",y3,[Oe("label",b3,[xr(Oe("input",{type:"checkbox","onUpdate:modelValue":L[17]||(L[17]=re=>M.value=re)},null,512),[[Ap,M.value]]),L[42]||(L[42]=Oe("span",null,"循环",-1))]),Oe("button",{class:"cinematic-close",onClick:L[18]||(L[18]=re=>B.value=!1),"aria-label":"收起运镜面板"}," × ")])]),Oe("div",M3,[Oe("button",{class:"archive-btn archive-btn--solid cinematic-primary",onClick:eu},Mn(ue.value),1),Oe("button",{class:"archive-btn archive-btn--ghost cinematic-secondary",onClick:L[19]||(L[19]=re=>$()),disabled:!E.value&&!b.value&&y.value===0}," 停止 ",8,C3)]),Oe("div",T3,[L[44]||(L[44]=Oe("span",null,"进度",-1)),Oe("span",null,Mn(Math.round(y.value*100))+"%",1)]),Oe("input",{class:"cinematic-progress",type:"range",value:y.value*100,min:"0",max:"100",step:"1",disabled:""},null,8,E3),Oe("div",w3,[L[45]||(L[45]=Oe("span",null,"速度",-1)),Oe("span",null,Mn(S.value.toFixed(2))+"x",1)]),xr(Oe("input",{class:"cinematic-speed",type:"range","onUpdate:modelValue":L[20]||(L[20]=re=>S.value=re),min:"0.25",max:"3",step:"0.05",onInput:nu},null,544),[[fa,S.value,void 0,{number:!0}]]),Oe("div",R3,[L[46]||(L[46]=Oe("span",null,"平滑",-1)),Oe("span",null,Mn(Math.round(C.value*100))+"%",1)]),xr(Oe("input",{class:"cinematic-speed",type:"range","onUpdate:modelValue":L[21]||(L[21]=re=>C.value=re),min:"0",max:"1",step:"0.05",onInput:gl},null,544),[[fa,C.value,void 0,{number:!0}]]),Oe("label",I3,[xr(Oe("input",{type:"checkbox","onUpdate:modelValue":L[22]||(L[22]=re=>P.value=re),onChange:gl},null,544),[[Ap,P.value]]),L[47]||(L[47]=Oe("span",null,"主体锁定",-1))])],32)):ai("",!0),g.value>0?(vn(),Cn("div",D3,"FPS "+Mn(g.value),1)):ai("",!0)])]),n.value?(vn(),Cn("div",P3,[...L[48]||(L[48]=[Oe("div",{class:"loading-card"},[Oe("div",{class:"loading-dot"}),Oe("div",{class:"loading-title"},"场景正在展开"),Oe("div",{class:"loading-copy"},"模型与参考镜头正在同步到工作台。")],-1)])])):ai("",!0),p.value?(vn(),Cn("div",F3,[Oe("div",L3,[L[49]||(L[49]=Oe("div",{class:"eyebrow"},"Load Failed",-1)),L[50]||(L[50]=Oe("div",{class:"error-title"},"模型未能正常打开",-1)),Oe("div",B3,Mn(p.value),1),Oe("button",{class:"archive-btn archive-btn--solid",onClick:L[28]||(L[28]=re=>Ue(_f(le),_f(ae),null))}," 重新载入 ")])])):ai("",!0),ai("",!0),m.value?(vn(),Cn("div",{key:3,class:"focal-settings-panel",onMousedown:L[31]||(L[31]=bt(()=>{},["stop"])),onTouchstart:L[32]||(L[32]=bt(()=>{},["stop"])),onTouchmove:L[33]||(L[33]=bt(()=>{},["stop"])),onTouchend:L[34]||(L[34]=bt(()=>{},["stop"])),onTouchcancel:L[35]||(L[35]=bt(()=>{},["stop"]))},[L[52]||(L[52]=Oe("div",{class:"eyebrow"},"Lens Control",-1)),L[53]||(L[53]=Oe("div",{class:"focal-title"},"镜头焦距",-1)),xr(Oe("input",{type:"range","onUpdate:modelValue":L[29]||(L[29]=re=>A.value=re),min:w.value,max:ne.value,step:"1",onInput:he},null,40,U3),[[fa,A.value,void 0,{number:!0}]]),Oe("div",O3,[xr(Oe("input",{class:"focal-number-input",type:"number","onUpdate:modelValue":L[30]||(L[30]=re=>A.value=re),min:w.value,max:ne.value,step:"1",onChange:he},null,40,N3),[[fa,A.value,void 0,{number:!0}]]),L[51]||(L[51]=Oe("span",null,"px",-1))]),Oe("div",z3,[Oe("span",null,"当前 FOV: "+Mn(_.value.toFixed(1))+"°",1)]),Oe("div",k3,[Oe("span",null,"当前焦距: "+Mn(v.value.toFixed(1))+" px",1)]),Oe("button",{class:"archive-btn archive-btn--solid focal-reset-btn",onClick:ie},"恢复拍摄焦距")],32)):ai("",!0),!N.value&&k.value.length>0?(vn(),Cn("div",{key:4,class:"camera-track",onMousedown:L[36]||(L[36]=bt(()=>{},["stop"])),onTouchstart:L[37]||(L[37]=bt(()=>{},["stop"])),onTouchmove:L[38]||(L[38]=bt(()=>{},["stop"])),onTouchend:L[39]||(L[39]=bt(()=>{},["stop"]))},[Oe("div",H3,[L[54]||(L[54]=Oe("div",{class:"eyebrow"},"Shot Strip",-1)),Oe("div",V3,Mn(l.value?"按当前检索结果排序":"优先显示已打标签镜头"),1)]),(vn(!0),Cn(Ki,null,yv(k.value,(re,Se)=>(vn(),Cn("div",{key:re.id,class:Or(["camera-btn",{active:f.value===se(re)}]),onClick:bt(Te=>D(re),["stop"])},[re.image_url?(vn(),Cn("img",{key:0,src:re.image_url,class:"btn-thumb"},null,8,W3)):ai("",!0),re.tag?(vn(),Cn("div",X3,[Oe("div",q3,Mn(re.tag),1)])):re.image_url?ai("",!0):(vn(),Cn("span",Y3,"未命名视角"))],10,G3))),128))],32)):ai("",!0),c.value?(vn(),Cn("div",{key:5,class:"reference-overlay",onClick:L[40]||(L[40]=re=>{c.value="",u.value=""})},[L[55]||(L[55]=Oe("div",{class:"eyebrow"},"Reference Still",-1)),L[56]||(L[56]=Oe("div",{class:"ref-title"},"参考原图",-1)),Oe("img",{src:c.value,class:"ref-img"},null,8,Q3),u.value?(vn(),Cn("div",K3,[Oe("span",j3,Mn(u.value),1)])):ai("",!0),d.value.fl_y?(vn(),Cn("div",$3,[Oe("span",Z3,"焦距: "+Mn(d.value.fl_y.toFixed(1))+" px",1),Oe("span",J3,"FOV: "+Mn((2*Math.atan(d.value.h/(2*d.value.fl_y))*(180/Math.PI)).toFixed(1))+"°",1),Oe("span",e2,"分辨率: "+Mn(d.value.w)+"x"+Mn(d.value.h),1)])):ai("",!0),L[57]||(L[57]=Oe("div",{class:"ref-hint"},"点击关闭对比",-1))])):ai("",!0)],32))}},u2=_3(c2,[["__scopeId","data-v-37e29009"]]),f2={__name:"App",setup(i){return(e,t)=>(vn(),Cn("main",null,[ss(u2)]))}};NA(f2).mount("#app");
