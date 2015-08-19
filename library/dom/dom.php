<?php

namespace library\dom;
use library\dom\elements\components\html;
use library\tree\leaf;
use library\tree\branch;
use library\dom\object;
use library\dom\elements\components\button;
use library\dom\elements\components\form;
use library\dom\elements\components\group;
use library\dom\elements\components\script;
use library\dom\elements\components\text;
use library\dom\elements\components\title;
use library\dom\elements\components\body;
use library\dom\elements\components\link;
use library\dom\elements\element;
use library\dom\elements\elementCollection;
use library\dom\elements\components\elementFactory;

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
    public static $Objecthash = array();

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
        echo self::$root->render();
        self::update();
    }
    public static function getDocument(){
        return self::$root;
    }
    public static function loadData($data){

        self::$root = $data->getValue();
        self::bft();
    }
    public static function getElementByTagName($tag){
        $collection = new elementCollection();
        $elements = self::$hash[$tag];
        $collection->addElements($elements);
        return $collection;
    }
    public static function update(){
        $_SESSION['api_data'] = serialize(self::$root->getNode());
    }
    public static function load(){
        $data = unserialize($_SESSION['api_data']);
        self::loadData($data);
    }

    

    public static function bft($node = false){
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
        }
        return $bft;
    }
    
    
}
