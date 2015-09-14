<?php

namespace library\dom;
use library\dom\elements\components\html;
use library\dom\elements\element;
use library\dom\elements\paired;
use library\dom\elements\elementCollection;
use library\dom\elements\components\elementFactory;
use library\event\listener;

/**
 * Description of dom
 *
 * @author durso
 */
class dom {
    protected static $hash = array();
    protected static $root;
    protected static $doctype = "<!DOCTYPE html>";
    protected static $tagName;
    protected static $listeners = array();


    public static function init(){

        self::$root = new html();
        self::addElement(self::$root);
    }
    public static function add($object){
        if(is_array($object)){
            foreach($object as $item){
                self::add($item);
            }
        } else {
            self::$root->addComponent($object);
            if($object instanceof element){
                self::bft();
            }
        }
    }
    public static function addElement(element $element){
        if($element->hasId()){
            self::$hash[$element->getId()] = $element;
        }
        self::$hash[$element->getTag()][] = $element;
    }
    
    public static function createElement($tag){
        return elementFactory::createByTag($tag);
    }
    public static function createTextNode($value){
        return elementFactory::createText($value);
    }

    public static function getElementById($id){
        return self::$hash[$id];
    }

    public static function setDoctype($doctype){
        self::$doctype = $doctype;
    }
    public static function save(){
        echo self::$doctype;
        $page = self::update();
        echo $page;
    }
    public static function getDocument(){
        return self::$root;
    }
    public static function loadData($data){
        $page = self::buildTree($_SESSION['api_data']['page'], false);
        self::$root = $page[0];
        self::$listeners = $data;
        self::bft(false, true);
    }
    public static function addEventListener(listener $listener,$id,$event){
        self::$listeners[$id][$event] = $listener;
    }
    public static function removeEventListener($id,$event){
        unset(self::$listeners[$id][$event]);
    }
    public static function getEventListener($id,$event){
        return self::$listeners[$id][$event];
    }

    public static function getElementByTagName($tag){
        $collection = new elementCollection();
        $elements = self::$hash[$tag];
        $collection->addElements($elements);
        return $collection;
    }
    public static function update(){
        $page =  self::$root->render();
        $_SESSION['api_data'] = array();
        $_SESSION['api_data']['page'] = $page;
        $_SESSION['api_data']['listeners'] = serialize(self::$listeners);
        return $page;
 

    }
    public static function load(){
        $data = unserialize($_SESSION['api_data']['listeners']);
        self::loadData($data);
    }

    public static function buildTree($string,$offsetTag){
        $pattern = '#(<[^!>]*[^\/][/]*>)#';
        $components = preg_split($pattern,$string,-1,PREG_SPLIT_DELIM_CAPTURE);
        $list = array();
        $offset = false;
        $collection = array();
        foreach($components as $key => $value){

            $value = trim($value);
            $len = strlen($value);
            if(!$len){
                continue;
            }
            if($value[0] == "<" && $value[1] != "!"){
                if($value[1] != "/"){
                    $pos = strpos($value, " ");
                    if($pos){
                        $tag = substr($value,1,$pos - 1);
                    } else {
                        $tag = substr($value,1,-1);
                    }
                    if($offsetTag && !$offset){
                        if($offsetTag != $tag){
                            continue;
                        }
                        $offset = true;
                    }

                    $element = elementFactory::createByTag($tag);
                } else {
                    array_pop($list);
                    continue;
                }
            } else {
                $element = elementFactory::createText($value);
            }
            if(empty($list)){      
                $collection[] = $element;
            } else {
                $parent = end($list);
                reset($list);
                $parent->addComponent($element);

            }
            if($element instanceof paired){
                $list[] = $element;
            }
            if($element instanceof element){
                if($pos){
                    $attr = substr($value,$pos,-1);
                    $element->stringToAttr($attr);
                }
            }
        }

        return $collection;
    }

    public static function bft($node = false,$render = false){
        if($node){
            $list = array($node);
        } else {
            $list = self::$root->getNode()->getChildren();
        }
        $bft = array();
        while(!empty($list)){
            $node = array_shift($list);
            $element = $node->getValue();
            if($node->hasChild()){
                foreach($node->getChildren() as $child){
                    if(!in_array($child,$list,true)){
                        $list[] = $child;
                    }
                }
            }
            $bft[] = $node;
            if($element instanceof element){
                self::addElement($element);
            }
            if($render){
                $element->setRenderFlag($render);
            }
        }
        return $bft;
    }
    
    
}
