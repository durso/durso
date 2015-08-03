<?php

namespace library\dom;
use library\dom\elements\html;
use library\dom\object;
use library\dom\elements\components\button;
use library\dom\elements\components\form;
use library\dom\elements\components\group;
use library\dom\elements\components\script;
use library\dom\elements\components\text;
use library\dom\elements\components\title;
use library\dom\components\template;
use library\dom\elements\layout;
use library\dom\elements\elementCollection;

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

    public static function init(){
        self::$root = new html();
    }
    public static function add(object $object){
        self::$root->addComponent($object);
    }
    public static function addObjectId(object $object){
        if($object->hasId()){
            self::$hash[$object->getId()] = $object;
        }			
    }
    public static function getElementById($id){
        return self::$hash[$id];
    }

    public static function setDoctype($doctype){
        self::$doctype = $doctype;
    }
    public static function save(){
        echo self::$doctype;
        echo self::$root->render();
        self::update();
    }
    public static function getDocument(){
        return self::$root;
    }
    public static function loadData($data){
        self::$root = $data;
        self::bft();
    }
    public static function getElementByTagName($tag){
        $collection = new elementCollection();
        $elements = self::$tagName[$tag];
        $collection->addElements($elements);
        return $collection;
    }
    public static function update(){
        $_SESSION['api_data'] = serialize(self::$root);
    }
    public static function load(){
        $data = unserialize($_SESSION['api_data']);
        self::loadData($data);
    }

    

    public static function bft(){
        $list = self::$root->getNode()->getChildren();
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
            if($element->hasId()){
                self::$hash[$element->getId()] = $element;
            }
            if($element->hasTag()){
                self::$tagName[$element->getTag()][] = $element;
            }
        }
        return $bft;
    }
    
    
}
